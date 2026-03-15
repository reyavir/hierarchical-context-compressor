from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import click
from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.tree import Tree

from .formatter import format_root_agents_md, format_root_llms, wrap_agents_md_header
from .tree import get_tree, read_files_by_paths


console = Console()

# --- Phase 1: Directory selection ---
DIRECTORY_SELECTION_SYSTEM = """You are given the full recursive file tree of a repository. Output a list of directory paths (one per line, relative to repo root) that should have their own AGENTS.md—e.g. substantial subprojects, docs, app source, scripts. Root must be included (use "." or empty for root). Output only paths, one per line, no explanation. Directories only, no files."""

# --- Phase 2: Discovery (which files to read per dir) ---
DISCOVERY_SYSTEM = """You are given the recursive file tree of one directory. Output a list of file paths (one per line, relative to this directory) that are essential to understand setup, commands, code style, and entry points (e.g. README, config, main source). Output only paths, one per line, no explanation."""

# --- Phase 3: Generation ---
ROOT_AGENTS_MD_SYSTEM_PROMPT = """You are writing the **root** repository index (agents.md).

**Structure:** (1) At most 2 short paragraphs describing what this codebase is and how it is structured—specific names, entrypoints, or paths only. (2) A table of contents: one bullet per directory with a link to its AGENTS.md and exactly one concrete phrase (e.g. "FastAPI API handlers (`app/api/*`)" or "Next.js frontend (`apps/web/`)"). Each bullet must include at least one path or filename in backticks. Total overview text (outside bullets) must be at most 5 sentences.

Do not use vague phrases like "well organized" or "follows best practices" without a concrete example. Base everything on the provided tree and file contents."""

AGENTS_MD_SYSTEM_PROMPT = """You are writing an Operational Manual for an AI agent for **this directory**. Extract USEFUL, SPECIFIC information only. Every bullet must reference a concrete file path, command, or symbol (in backticks). No broad or vague statements.

**## Setup & Commands**
- List ONLY commands you can point to in real files (e.g. package.json scripts, Makefile, pyproject.toml). For each: give the **exact command** and **source** (e.g. `npm run test` from `package.json` scripts). If you find no runnable commands, say exactly: "No runnable commands detected in this directory." Do not guess or invent commands.

**## Code Style & Patterns**
- Mention ONLY patterns you can back with a concrete example: file name, function/class name, or config key. Include at least one concrete path or symbol per bullet (e.g. `src/api/user.py:get_user`, `tests/test_user.py`). Do not write vague lines like "follows good patterns" or "uses modern practices" unless you add a specific file or symbol right after.

**## Implementation Details**
- Name **specific entrypoints and modules** (e.g. `main.py`, `app.py`, `src/index.tsx`) and state which file to open first for a given kind of change. Focus on how code is wired: imports, module boundaries, layers. Do not restate the README.

**Banned:** Do not use generic phrases ("well structured", "best practices", "clean code") without immediately following with a concrete example (file path or symbol in backticks). If a bullet has no code/path reference, omit it.

Output exactly the three level-2 headers above. Be concise; bullets over paragraphs."""

# Caps
MAX_DIRS_PHASE1 = 15
MAX_PATHS_DISCOVERY = 20
MAX_CHARS_PER_FILE = 8000
MAX_TOTAL_CHARS_DISCOVERY = 40000


def is_web_app(repo_root: Path) -> bool:
    """
    Heuristic: treat repo as a web app if it has web-facing entrypoints or standard web layout.
    Uses path listing only (no crawler).
    """
    repo_root = repo_root.resolve()
    try:
        names = {p.name.lower() for p in repo_root.iterdir()}
    except OSError:
        return False
    if names & {"templates", "static", "public", "app", "src"}:
        return True
    if names & {"index.html", "app.py", "main.py", "server.py", "app.js", "index.js"}:
        if "package.json" in names or "requirements.txt" in names:
            return True
    if "package.json" in names:
        try:
            data = json.loads((repo_root / "package.json").read_text(encoding="utf-8", errors="replace"))
            scripts = data.get("scripts", {}) or {}
            if any(k in scripts for k in ("start", "dev", "serve")):
                return True
        except Exception:
            pass
    for name in ("requirements.txt", "pyproject.toml"):
        p = repo_root / name
        if not p.exists():
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace").lower()
            if any(fw in text for fw in ("flask", "django", "fastapi", "starlette")):
                return True
        except Exception:
            pass
    return False


def _get_openai_client(base_url: Optional[str] = None) -> Optional[OpenAI]:
    project_root = Path(__file__).resolve().parents[1]
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=api_key)


def _parse_directory_selection(completion: str, repo_root: Path) -> List[Path]:
    """Parse phase 1 response into list of Path; root always included; only existing dirs."""
    seen: set[str] = set()
    result: List[Path] = []
    for line in completion.strip().splitlines():
        rel = line.strip().strip("/").strip() or "."
        if rel in seen:
            continue
        if rel == "." or rel == "":
            result.insert(0, repo_root)
            seen.add(".")
            continue
        path = (repo_root / rel).resolve()
        try:
            path.relative_to(repo_root)
        except ValueError:
            continue
        if path.is_dir() and path not in result:
            result.append(path)
            seen.add(rel)
    if repo_root not in result:
        result.insert(0, repo_root)
    return result[:MAX_DIRS_PHASE1]


def _parse_discovery_paths(completion: str, dir_path: Path) -> List[str]:
    """Parse phase 2 response into list of relative file paths that exist under dir_path."""
    result: List[str] = []
    for line in completion.strip().splitlines():
        rel = line.strip().lstrip("/").strip()
        if not rel:
            continue
        path = (dir_path / rel).resolve()
        try:
            path.relative_to(dir_path)
        except ValueError:
            continue
        if path.is_file():
            result.append(rel)
    return result[:MAX_PATHS_DISCOVERY]


def run_phase1_directory_selection(
    client: OpenAI,
    discovery_model: str,
    repo_root: Path,
) -> List[Path]:
    """Phase 1: one LLM call with full tree -> list of dirs that get AGENTS.md."""
    tree_text = get_tree(repo_root)
    try:
        resp = client.chat.completions.create(
            model=discovery_model,
            messages=[
                {"role": "system", "content": DIRECTORY_SELECTION_SYSTEM},
                {"role": "user", "content": f"Repository tree:\n\n```\n{tree_text}\n```"},
            ],
            max_tokens=1024,
        )
        content = (resp.choices[0].message.content or "").strip()
        if content:
            return _parse_directory_selection(content, repo_root)
    except Exception as e:
        console.print(f"[yellow]Phase 1 (directory selection) failed: {e}[/yellow]")
    return [repo_root]


def run_phase2_discovery(
    client: OpenAI,
    discovery_model: str,
    dir_path: Path,
) -> Dict[str, str]:
    """Phase 2: discovery call for this dir -> file paths -> read_files_by_paths. Minimal fallback: README.md if present."""
    tree_text = get_tree(dir_path)
    paths: List[str] = []
    try:
        resp = client.chat.completions.create(
            model=discovery_model,
            messages=[
                {"role": "system", "content": DISCOVERY_SYSTEM},
                {"role": "user", "content": f"Directory tree:\n\n```\n{tree_text}\n```"},
            ],
            max_tokens=1024,
        )
        content = (resp.choices[0].message.content or "").strip()
        if content:
            paths = _parse_discovery_paths(content, dir_path)
    except Exception as e:
        console.print(f"[yellow]Discovery failed for {dir_path}: {e}[/yellow]")
    if not paths and (dir_path / "README.md").is_file():
        paths = ["README.md"]
    return read_files_by_paths(
        dir_path,
        paths,
        max_chars_per_file=MAX_CHARS_PER_FILE,
        max_total_chars=MAX_TOTAL_CHARS_DISCOVERY,
    )


def _trim_and_tag_snippets(discovered_contents: Dict[str, str]) -> Dict[str, str]:
    """
    Trim known file types to the most relevant parts and tag them for the LLM.
    Returns path -> content (possibly trimmed and with a role label in the key).
    """
    result: Dict[str, str] = {}
    for path, content in discovered_contents.items():
        path_lower = path.lower()
        if path_lower == "package.json":
            try:
                data = json.loads(content.split("... (truncated)")[0])
                scripts = data.get("scripts") or {}
                if scripts:
                    result["package.json (scripts)"] = json.dumps({"scripts": scripts}, indent=2)
                else:
                    result[path] = content[:2000]
            except Exception:
                result[path] = content[:4000]
        elif path_lower == "pyproject.toml":
            out: List[str] = []
            in_project = in_tool = False
            for line in content.splitlines():
                if line.strip().startswith("[project]"):
                    in_project = True
                    in_tool = False
                elif line.strip().startswith("[tool."):
                    in_tool = True
                    in_project = False
                elif line.strip().startswith("[") and "]" in line:
                    in_project = in_tool = False
                if in_project or in_tool:
                    out.append(line)
            result["pyproject.toml (config)"] = "\n".join(out)[:3000] if out else content[:3000]
        elif path_lower.endswith(("main.py", "app.py", "index.py", "index.js", "index.ts", "index.tsx")):
            lines = content.splitlines()
            if len(lines) > 80:
                result[f"{path} (entrypoint, first 80 lines)"] = "\n".join(lines[:80]) + "\n... (truncated)"
            else:
                result[f"{path} (entrypoint)"] = content
        else:
            result[path] = content
    return result


def _build_user_message(dir_path: Path, repo_root: Path, tree_text: str, discovered_contents: Dict[str, str]) -> str:
    rel = dir_path.relative_to(repo_root).as_posix() if dir_path != repo_root else "."
    parts = [f"Directory: `{rel}`", "", "Recursive file tree:", "```", tree_text, "```", ""]
    if discovered_contents:
        trimmed = _trim_and_tag_snippets(discovered_contents)
        parts.append("File contents (use these to extract exact commands, paths, and symbols):")
        keys = set(trimmed.keys())
        if any("package.json" in k for k in keys):
            parts.append("From package.json scripts, list only commands that CI or a dev would run (build, test, lint, dev, start).")
        if any("pyproject.toml" in k for k in keys):
            parts.append("From pyproject.toml, infer package name and test/lint tools if present.")
        parts.append("")
        for path, content in trimmed.items():
            parts.append(f"--- {path} ---")
            parts.append(content)
            parts.append("")
    return "\n".join(parts)


# Phrases that make a bullet likely generic fluff if there's no code/path reference
_GENERIC_PHRASES = re.compile(
    r"\b(best practices|well structured|clean code|modern practices|good patterns|"
    r"follows conventions|properly organized|maintainable|readable code)\b",
    re.IGNORECASE,
)


def _drop_generic_bullets(md: str) -> str:
    """
    Remove list items that have no code/path reference (no backticks) and contain generic fluff.
    Keeps bullets that mention at least one `path`, `symbol`, or `command`.
    """
    lines = md.split("\n")
    out: List[str] = []
    for line in lines:
        stripped = line.strip()
        is_bullet = stripped.startswith("- ") or stripped.startswith("* ")
        if not is_bullet:
            out.append(line)
            continue
        has_backtick = "`" in line
        if has_backtick:
            out.append(line)
            continue
        if _GENERIC_PHRASES.search(line):
            continue
        out.append(line)
    return "\n".join(out)


def _ensure_three_sections(md: str) -> str:
    required = ["## Setup & Commands", "## Code Style & Patterns", "## Implementation Details"]
    seen = [h in md for h in required]
    if all(seen):
        return md
    out = md.rstrip()
    for i, h in enumerate(required):
        if not seen[i]:
            out += "\n\n" + h + "\n\n_No content generated for this section._"
    return out


def generate_agents_md_with_llm(
    client: Optional[OpenAI],
    generation_model: str,
    dir_path: Path,
    repo_root: Path,
    tree_text: str,
    discovered_contents: Dict[str, str],
    is_root: bool,
) -> str:
    """
    Generate AGENTS.md content for one directory. Root uses ToC prompt; nested uses three-section prompt.
    Returns full document including header (wrap_agents_md_header applied).
    """
    rel = dir_path.relative_to(repo_root).as_posix() if dir_path != repo_root else "."

    if client is None:
        body = "## Table of contents\n\n_No LLM available._\n\n## Setup & Commands\n\n_No content._\n\n## Code Style & Patterns\n\n_No content._\n\n## Implementation Details\n\n_No content._"
        if not is_root:
            body = "## Setup & Commands\n\nNo scripts or makefiles detected.\n\n## Code Style & Patterns\n\n_Infer from file tree._\n\n## Implementation Details\n\n_No content._"
        return wrap_agents_md_header(dir_path, repo_root, body)

    user_content = _build_user_message(dir_path, repo_root, tree_text, discovered_contents)
    system_prompt = ROOT_AGENTS_MD_SYSTEM_PROMPT if is_root else AGENTS_MD_SYSTEM_PROMPT
    try:
        resp = client.chat.completions.create(
            model=generation_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            max_tokens=4096,
        )
        content = (resp.choices[0].message.content or "").strip()
        if not content:
            raise ValueError("Empty response")
        content = _drop_generic_bullets(content)
        if not is_root:
            content = _ensure_three_sections(content)
        return wrap_agents_md_header(dir_path, repo_root, content)
    except Exception as exc:
        console.print(f"[yellow]Generation failed for {rel}: {exc}[/yellow]")
        body = "## Setup & Commands\n\n_Generation failed._\n\n## Code Style & Patterns\n\n_No content._\n\n## Implementation Details\n\n_No content._"
        if is_root:
            body = "## Table of contents\n\n_Generation failed._"
        return wrap_agents_md_header(dir_path, repo_root, body)


def build_agents_md_contents(
    repo_root: Path,
    selected_dirs: List[Path],
    client: Optional[OpenAI],
    discovery_model: str,
    generation_model: str,
) -> tuple[Dict[Path, str], Dict[Path, str]]:
    """
    Run phase 2 and 3 for each selected dir. Returns (agents_md_contents, folder_summaries).
    folder_summaries: one-line summary per dir for root ToC (we use first line of generated content or placeholder).
    """
    contents: Dict[Path, str] = {}
    summaries: Dict[Path, str] = {}
    for dir_path in selected_dirs:
        is_root = dir_path.resolve() == repo_root.resolve()
        discovered = run_phase2_discovery(client, discovery_model, dir_path)
        tree_text = get_tree(dir_path)
        content = generate_agents_md_with_llm(
            client,
            generation_model,
            dir_path,
            repo_root,
            tree_text,
            discovered,
            is_root=is_root,
        )
        contents[dir_path] = content
        first_line = content.split("\n")[2] if "\n" in content else content[:80]
        summaries[dir_path] = first_line.replace("#", "").strip()[:120]
    return contents, summaries


def render_tree(repo_root: Path, selected_dirs: List[Path], agents_md_contents: Dict[Path, str]) -> None:
    """Rich tree view for dry-run."""
    tree = Tree("[bold]Repository[/bold]")
    for dir_path in selected_dirs:
        rel = dir_path.relative_to(repo_root).as_posix() if dir_path != repo_root else "."
        label = f"[bold]{rel}[/bold]" + (" [AGENTS.md]" if dir_path in agents_md_contents else "")
        tree.add(label)
    console.print(tree)
    console.print()
    console.print("[bold magenta]AGENTS.md generated for[/bold magenta]")
    for path in agents_md_contents:
        rel = path.relative_to(repo_root).as_posix() if path != repo_root else "."
        console.print(f"  [bold]{rel}[/bold] -> AGENTS.md")


def write_context_files(
    repo_root: Path,
    selected_dirs: List[Path],
    agents_md_contents: Dict[Path, str],
    folder_summaries: Dict[Path, str],
) -> None:
    agents_toc_path = repo_root / "agents.md"
    agents_toc_path.write_text(
        format_root_agents_md(repo_root, selected_dirs, folder_summaries),
        encoding="utf-8",
    )
    if is_web_app(repo_root):
        llms_path = repo_root / "llms.txt"
        llms_path.write_text(
            format_root_llms(repo_root, selected_dirs, folder_summaries),
            encoding="utf-8",
        )
    for dir_path, content in agents_md_contents.items():
        if dir_path.resolve() == repo_root.resolve():
            continue
        agents_path = dir_path / "AGENTS.md"
        agents_path.write_text(
            _merge_with_existing_agents_md(agents_path, content),
            encoding="utf-8",
        )


def _merge_with_existing_agents_md(path: Path, regenerated: str) -> str:
    if not path.exists():
        return regenerated
    try:
        existing = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return regenerated
    m = re.search(r"^(#{2,3})\s+Rules\s*$", existing, re.MULTILINE)
    if not m:
        return regenerated
    rules_block = existing[m.start() :].rstrip()
    return regenerated.rstrip() + "\n\n" + rules_block + "\n"


@click.command()
@click.option(
    "--root",
    "root_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("."),
    help="Root directory of the repository to analyze.",
)
@click.option(
    "--discovery-model",
    type=str,
    default="gpt-4o-mini",
    show_default=True,
    help="OpenAI model for directory selection and file discovery (default gpt-4o-mini).",
)
@click.option(
    "--generation-model",
    type=str,
    default="gpt-4o",
    show_default=True,
    help="OpenAI model for AGENTS.md generation (default gpt-4o).",
)
@click.option(
    "--model",
    type=str,
    default=None,
    help="Use this model for both discovery and generation (overrides --discovery-model and --generation-model).",
)
@click.option(
    "--base-url",
    type=str,
    default=None,
    help="Custom API base URL (e.g. LiteLLM proxy). Or set OPENAI_BASE_URL.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Do not write files; print a tree view of the computed context.",
)
def cli(
    root_dir: Path,
    discovery_model: str,
    generation_model: str,
    model: Optional[str],
    base_url: Optional[str],
    dry_run: bool,
) -> None:
    """Generate a hierarchical context map for a codebase (agents.md + AGENTS.md)."""
    if model:
        discovery_model = generation_model = model
    base_url = base_url or os.getenv("OPENAI_BASE_URL")
    repo_root = root_dir.resolve()
    console.print(f"[bold]Scanning repository:[/bold] {repo_root}")

    client = _get_openai_client(base_url)
    selected_dirs = run_phase1_directory_selection(client, discovery_model, repo_root)
    if not client:
        selected_dirs = [repo_root]
    console.print(f"[dim]Selected {len(selected_dirs)} directories for AGENTS.md[/dim]")

    agents_md_contents, folder_summaries = build_agents_md_contents(
        repo_root, selected_dirs, client, discovery_model, generation_model
    )

    if dry_run:
        console.print("[bold blue]Dry run - not writing any files.[/bold blue]")
        render_tree(repo_root, selected_dirs, agents_md_contents)
    else:
        console.print("[bold green]Writing agents.md and AGENTS.md files...[/bold green]")
        write_context_files(repo_root, selected_dirs, agents_md_contents, folder_summaries)
        console.print("[bold green]Done.[/bold green]")


if __name__ == "__main__":
    cli()
