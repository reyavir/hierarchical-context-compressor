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
DIRECTORY_SELECTION_SYSTEM = """You are given the full recursive file tree of a repository. Output a list of directory paths (one per line, relative to repo root) that should have their own AGENTS.md. Root (".") must always be included.

Select ONLY directories where an AI coding agent would actively write or modify code, config, or scripts:
- Source code directories (app logic, services, modules, libraries)
- CLI entry points and scripts
- Infrastructure and build tooling directories
- Configuration directories (if non-trivial)

Do NOT select:
- Directories containing only documentation, notes, or markdown files with no runnable code
- Task-tracking, changelog, or archive directories
- Data, fixture, or sample-data directories
- Example or demo directories
- Cache, build output, or generated artifact directories

Output only paths, one per line, no explanation. Directories only, no files."""

# --- Phase 2: Discovery (which files to read per dir) ---
DISCOVERY_SYSTEM = """You are given the recursive file tree of one directory. Output a list of file paths (one per line, relative to this directory) that are essential to understand setup, commands, code style, and entry points (e.g. README, config, main source). Output only paths, one per line, no explanation."""

# --- Phase 3: Generation ---
ROOT_AGENTS_MD_SYSTEM_PROMPT = """You are writing the **root** repository index (agents.md) for an AI coding agent.

**Purpose:** Route the agent to the correct folder before it edits code.

**Required output:**
1) 1-2 short imperative lines at the top that say:
- Start here, then navigate to the closest directory AGENTS.md before coding.
- Do not operate from root unless necessary.
2) A table of contents with one bullet per selected directory. Each bullet must include:
- link to that folder's `AGENTS.md`
- exactly one concrete phrase with at least one backticked path/symbol.

Keep overview text outside bullets to at most 5 short sentences—no wrap-up or "in conclusion" lines.
No fenced code blocks (```). No sample code; the agent has the repo. Inline `paths` only.
Do not use vague phrases like "well organized" or "best practices". Base everything on the provided tree and file contents."""

AGENTS_MD_SYSTEM_PROMPT = """You are writing an Operational Manual for an AI coding agent for **this directory**.
Tell the agent how to act in this folder—minimal text, maximum signal. Do not describe the codebase at length.

Hard requirements:
- Scope strictly to this directory; do not reference unrelated paths outside it unless the user message proves a necessary link.
- Imperative only: "Do X in `path/file`" / "Run `command`".
- Every bullet must include a concrete path, command, or symbol in backticks.
- Prefer bullets over paragraphs. No long explanations, no repeated phrasing, no "for example" essays.
- **Never** use fenced code blocks (```) or multi-line code samples. The agent can read the repo—do not paste `const`/`function` snippets. Inline `backticks` for files/symbols/commands only.
- For each common task playbook (e.g. add route/service/test): at most **3-4 bullets total** for that task. Merge redundant steps (do not use separate "Create file / Define logic / Example" unless each line adds unique, cited detail).
- **Banned closing fluff:** no summary paragraphs, no "By maintaining these guidelines…", "In summary…", or similar meta sign-off. End on the last factual bullet; stop.

You MAY use these level-2 headers only when you have evidence-backed content:

## Setup & Commands
- Exact runnable commands and source (e.g. `npm run test` from `package.json`). Scoped variants if real. Omit if none.

## Code Style & Patterns
- Must-follow rules for edits here, with backticked file/symbol proof. Omit if none.

## Implementation Details
- Entry points and control files in **this** folder (routers, bootstrap, config) in one tight bullet list.
- Short playbooks only: 3-4 bullets per task, each bullet one line, imperative. Omit if unsupported.

Use at most the three headers above. Leave out weak sections entirely."""

# --- Directory type classification and type-specific prompts ---
# Type -> full system prompt for generation (persona + boundary + what to extract)
TYPE_SYSTEM_PROMPTS: Dict[str, str] = {
    "core": """You are writing an Operational Manual for a **Senior Lead Engineer** in this directory (core/source code). Actionable edits only—dense bullets, no prose essays. Every bullet must include a concrete path, command, or symbol (in backticks).

**Boundary:** This agent owns production code. Preserve exported APIs and architectural contracts; call out breaking impact briefly if needed—one line, backticked.

**Format:** No ``` fences or pasted source. Playbooks ≤3-4 bullets per task (add route/service/test). No closing summaries or guideline wrap-ups.

You MAY use these level-2 headers only when you have evidence-backed content:
## Setup & Commands — Exact build/run/test commands from real files.
## Code Style & Patterns — Must-follow rules with backticked proof.
## Implementation Details — Entrypoints + compressed playbooks (3-4 bullets each).

Stay folder-scoped; omit weak sections.""",

    "docs": """You are writing an Operational Manual for a **Technical Writer** in this directory (documentation). Dense imperative bullets only. Every bullet must reference a concrete path or command (in backticks).

**Boundary:** Only write in Markdown; do not suggest application source edits.

**Format:** No ``` fences or long prose. Playbooks ≤3-4 bullets per task (new page/section/link). No closing wrap-up paragraphs.

You MAY use these level-2 headers only when you have evidence-backed content:
## Setup & Commands — Exact build/serve/lint/link-check commands.
## Code Style & Patterns — Must-follow doc rules with backticked proof.
## Implementation Details — Entry docs/config + compressed playbooks.

Stay folder-scoped; omit weak sections.""",

    "tests": """You are writing an Operational Manual for a **QA Engineer** in this directory (tests). Minimal text; how to run and where to add tests. Every bullet must reference a concrete path or command (in backticks).

**Boundary:** Never modify production source to pass tests; tests/fixtures/config only.

**Format:** No ``` fences or pasted test code. Playbooks ≤3-4 bullets per task (new case/fixture/suite). No meta sign-offs at the end.

You MAY use these level-2 headers only when you have evidence-backed content:
## Setup & Commands — Exact test commands with scopes/filters when real.
## Code Style & Patterns — Must-follow naming/mocking/fixture rules, backticked.
## Implementation Details — Entry tests + compressed playbooks.

Stay folder-scoped; omit weak sections.""",

    "infra": """You are writing an Operational Manual for a **DevOps Specialist** in this directory (scripts/infra). Imperative, minimal; env vars, order, commands. Every bullet must reference a concrete path or command (in backticks).

**Boundary:** No application source edits—scripts, Docker, CI, Terraform only. Destructive ops: one backticked warning line.

**Format:** No ``` fences. Playbooks ≤3-4 bullets per task. No closing essays.

You MAY use these level-2 headers only when you have evidence-backed content:
## Setup & Commands — Exact commands + required env vars.
## Code Style & Patterns — Must-follow conventions, backticked.
## Implementation Details — Entry files + compressed playbooks (job/env/rollout).

Stay folder-scoped; omit weak sections.""",

    "generic": """You are writing an Operational Manual for an AI coding agent in this directory. No persona—only explicit, folder-scoped actions. Every bullet must reference a concrete file path, command, or symbol (in backticks).

**Format:** No ``` fences or pasted code. Bullets only where possible. Playbooks ≤3-4 bullets per task. No summary/closing fluff.

You MAY use these level-2 headers only when you have evidence-backed content:
## Setup & Commands — Exact commands from real files.
## Code Style & Patterns — Must-follow constraints, backticked.
## Implementation Details — Entrypoints + compressed playbooks.

Stay folder-scoped; avoid generic phrases; omit weak sections.""",
}


def _get_system_prompt_for_type(type_key: str, templates_dir: Optional[Path] = None) -> str:
    """
    Return the system prompt for a directory type. If templates_dir is set and contains
    {type_key}.md or {type_key}.txt, use that file's content; otherwise use TYPE_SYSTEM_PROMPTS.
    Empty template file content falls back to built-in prompt.
    """
    if templates_dir is not None and templates_dir.is_dir():
        for ext in (".md", ".txt"):
            path = templates_dir / f"{type_key}{ext}"
            if path.is_file():
                content = path.read_text(encoding="utf-8", errors="replace").strip()
                if content:
                    return content
    return TYPE_SYSTEM_PROMPTS.get(type_key, AGENTS_MD_SYSTEM_PROMPT)


def _classify_directory(dir_path: Path, repo_root: Path) -> str:
    """
    Classify directory into type based on path and contents. Returns type_key.
    Order: infra -> docs -> tests -> core -> generic.
    """
    try:
        rel_parts = dir_path.resolve().relative_to(repo_root.resolve()).parts
    except ValueError:
        rel_parts = ()
    path_lower = "/".join(rel_parts).lower()
    try:
        entries = list(dir_path.iterdir())
        names = {p.name.lower() for p in entries}
        files = [p.name.lower() for p in entries if p.is_file()]
    except OSError:
        names = set()
        files = []

    # INFRA: path has scripts, infra, terraform; or has Dockerfile, .yml
    if any(p in path_lower for p in ("scripts", "infra", "terraform", ".github")):
        return "infra"
    if "dockerfile" in names or any(f.endswith(".yml") or f.endswith(".yaml") for f in files):
        return "infra"

    # DOCS: path has docs; or directory is mostly .md files
    if "docs" in path_lower:
        return "docs"
    md_count = sum(1 for f in files if f.endswith(".md"))
    if md_count >= 2 and len(files) <= 10:
        return "docs"

    # TESTS: path has tests, test, spec; or files like test_*.py, *.spec.*
    if any(p in path_lower for p in ("tests", "test", "spec", "__tests__")):
        return "tests"
    if any(f.startswith("test_") or ".spec." in f for f in files):
        return "tests"

    # CORE: only if strong signal—path is a known source root or dir has multiple source files
    source_roots = ("src", "app", "lib", "core", "packages", "pkg")
    if any(p in path_lower for p in source_roots):
        return "core"
    source_extensions = (".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".rs", ".java")
    source_count = sum(1 for f in files if any(f.endswith(ext) for ext in source_extensions))
    if source_count >= 2:
        return "core"

    # GENERIC: fallback when nothing clearly fits (misc folders, config-only, single file, etc.)
    return "generic"


# Caps
MAX_DIRS_PHASE1 = 15
MAX_PATHS_DISCOVERY = 20
MAX_CHARS_PER_FILE = 8000
MAX_TOTAL_CHARS_DISCOVERY = 40000
# Max lines for generated AGENTS.md **body** (before `### Local Agent Context` header). ~100 lines keeps files skimmable.
MAX_AGENTS_MD_BODY_LINES = 100


def is_web_app(repo_root: Path) -> bool:
    """
    Heuristic: treat repo as a web app only if it looks like a *user-facing* web app
    (something you'd point an llms.txt crawler at), not a backend API alone.
    - Requires evidence of a UI: templates/ static/ public/ or index.html, or a frontend stack in package.json.
    - Backend-only repos (e.g. FastAPI/Flask API with no templates/static) do NOT get llms.txt.
    """
    repo_root = repo_root.resolve()
    try:
        names = {p.name.lower() for p in repo_root.iterdir()}
    except OSError:
        return False
    # Clear UI signals: server-rendered or static assets
    if names & {"templates", "static", "public"}:
        return True
    # Single-page or static site at root
    if "index.html" in names:
        return True
    # Frontend app: package.json with start/dev/serve AND a known frontend framework in deps
    if "package.json" in names:
        try:
            data = json.loads((repo_root / "package.json").read_text(encoding="utf-8", errors="replace"))
            scripts = data.get("scripts", {}) or {}
            if not any(k in scripts for k in ("start", "dev", "serve")):
                return False
            deps = {k.lower() for k in (data.get("dependencies") or {}).keys()}
            deps |= {k.lower() for k in (data.get("devDependencies") or {}).keys()}
            frontend = {"next", "react-scripts", "vite", "vue", "nuxt", "angular", "remix", "svelte", "parcel"}
            if deps & frontend:
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
    client: Optional[OpenAI],
    discovery_model: str,
    repo_root: Path,
) -> List[Path]:
    """Phase 1: one LLM call with full tree -> list of dirs that get AGENTS.md."""
    if client is None:
        return [repo_root]
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
    client: Optional[OpenAI],
    discovery_model: str,
    dir_path: Path,
) -> Dict[str, str]:
    """Phase 2: discovery call for this dir -> file paths -> read_files_by_paths. Minimal fallback: README.md if present."""
    if client is None:
        paths: List[str] = ["README.md"] if (dir_path / "README.md").is_file() else []
        return read_files_by_paths(
            dir_path,
            paths,
            max_chars_per_file=MAX_CHARS_PER_FILE,
            max_total_chars=MAX_TOTAL_CHARS_DISCOVERY,
        )
    tree_text = get_tree(dir_path)
    paths = []
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


def _build_user_message(dir_path: Path, repo_root: Path, tree_text: str, discovered_contents: Dict[str, str]) -> str:
    rel = dir_path.relative_to(repo_root).as_posix() if dir_path != repo_root else "."
    parts = [f"Directory: `{rel}`", "", "Recursive file tree:", "```", tree_text, "```", ""]
    if discovered_contents:
        parts.append("Key file contents:")
        for path, content in discovered_contents.items():
            parts.append(f"--- {path} ---")
            parts.append(content)
            parts.append("")
    return "\n".join(parts)


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


def _prune_sections(md: str) -> str:
    """
    Drop whole sections (## Heading ...) that don't provide a unique signal after
    generic bullets are removed. A section is kept if it has either:
    - at least one backticked reference, or
    - at least 2 non-empty lines of content.
    """
    lines = md.split("\n")
    n = len(lines)
    # Find indices of level-2 headings
    heading_idxs: List[int] = [i for i, line in enumerate(lines) if line.startswith("## ")]
    if not heading_idxs:
        return md

    new_lines: List[str] = []
    # Keep everything before the first heading as-is
    first_idx = heading_idxs[0]
    new_lines.extend(lines[:first_idx])

    for idx, h_start in enumerate(heading_idxs):
        h_end = heading_idxs[idx + 1] if idx + 1 < len(heading_idxs) else n
        heading_line = lines[h_start]
        body_lines = lines[h_start + 1 : h_end]
        body_text = "\n".join(body_lines).strip()
        if not body_text:
            # Empty body: drop this section
            continue
        # Evaluate signal
        has_backtick = "`" in body_text
        nonempty = [l for l in body_lines if l.strip()]
        has_substance = len(nonempty) >= 2
        if not has_backtick and not has_substance:
            # No concrete reference and basically nothing there: drop
            continue
        # Keep section
        new_lines.append(heading_line)
        new_lines.extend(body_lines)

    return "\n".join(new_lines).rstrip()


def _limit_agents_md_body_lines(md: str, max_lines: int = MAX_AGENTS_MD_BODY_LINES) -> str:
    """Cap markdown body length (line count). Header is applied later by wrap_agents_md_header."""
    lines = md.split("\n")
    if len(lines) <= max_lines:
        return md
    kept = lines[:max_lines]
    return "\n".join(kept).rstrip() + f"\n\n_… (truncated to {max_lines} lines)_\n"


def generate_agents_md_with_llm(
    client: Optional[OpenAI],
    generation_model: str,
    dir_path: Path,
    repo_root: Path,
    tree_text: str,
    discovered_contents: Dict[str, str],
    is_root: bool,
    templates_dir: Optional[Path] = None,
    max_lines: int = MAX_AGENTS_MD_BODY_LINES,
) -> str:
    """
    Generate AGENTS.md content for one directory. Root uses ToC prompt; nested uses three-section prompt.
    Returns full document including header (wrap_agents_md_header applied).
    """
    rel = dir_path.relative_to(repo_root).as_posix() if dir_path != repo_root else "."

    # Type-aware prompt for non-root; optional user templates override built-in
    if is_root:
        system_prompt = ROOT_AGENTS_MD_SYSTEM_PROMPT
    else:
        type_key = _classify_directory(dir_path, repo_root)
        system_prompt = _get_system_prompt_for_type(type_key, templates_dir)

    def _wrap(body: str) -> str:
        return wrap_agents_md_header(dir_path, repo_root, body)

    if client is None:
        body = "## Table of contents\n\n_No LLM available._"
        if not is_root:
            body = "## Setup & Commands\n\nNo scripts or makefiles detected."
        return _wrap(body)

    user_content = _build_user_message(dir_path, repo_root, tree_text, discovered_contents)
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
        content = _prune_sections(content)
        content = _limit_agents_md_body_lines(content, max_lines)
        return _wrap(content)
    except Exception as exc:
        console.print(f"[yellow]Generation failed for {rel}: {exc}[/yellow]")
        body = "## Table of contents\n\n_Generation failed._" if is_root else "## Setup & Commands\n\n_Generation failed._"
        return _wrap(body)


def build_agents_md_contents(
    repo_root: Path,
    selected_dirs: List[Path],
    client: Optional[OpenAI],
    discovery_model: str,
    generation_model: str,
    templates_dir: Optional[Path] = None,
    max_lines: int = MAX_AGENTS_MD_BODY_LINES,
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
            templates_dir=templates_dir,
            max_lines=max_lines,
        )
        contents[dir_path] = content
        summaries[dir_path] = _extract_summary(content)
    return contents, summaries


def _extract_summary(content: str, max_len: int = 120) -> str:
    """
    Extract a one-line summary for the ToC from generated AGENTS.md content.
    Prefer the first substantive line (paragraph or bullet), not a ## section header.
    """
    lines = content.split("\n")
    for line in lines:
        s = line.strip()
        if not s:
            continue
        if s.startswith("### Local Agent Context"):
            continue
        if s.startswith("## "):
            continue
        return s.replace("#", "").strip()[:max_len]
    # Fallback: skip headers and single markdown link lines; prefer line that contains a backtick
    candidates: List[str] = []
    for line in lines:
        s = line.strip()
        if not s:
            continue
        if s.startswith("## "):
            continue
        if s.startswith("- [") and "]" in s[3:] and ("(" in s or ")" in s):
            continue
        candidates.append(s.replace("#", "").strip()[:max_len])
    if candidates:
        for c in candidates:
            if "`" in c:
                return c
        return candidates[0]
    return content.strip()[:max_len] if content else ""


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
            agents_path = repo_root / "AGENTS.md"
        else:
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
    m = re.search(r"^(#{2,3})\s+Rules\s*$", existing, re.MULTILINE | re.IGNORECASE)
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
@click.option(
    "--templates-dir",
    "templates_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Directory with optional per-type templates: docs.md, tests.md, core.md, infra.md, generic.md. If present, used as the system prompt for that directory type.",
)
@click.option(
    "--max-lines",
    "max_lines",
    type=int,
    default=MAX_AGENTS_MD_BODY_LINES,
    show_default=True,
    help="Maximum number of lines per generated AGENTS.md body. Increase for large or complex directories.",
)
def cli(
    root_dir: Path,
    discovery_model: str,
    generation_model: str,
    model: Optional[str],
    base_url: Optional[str],
    dry_run: bool,
    templates_dir: Optional[Path],
    max_lines: int,
) -> None:
    """Generate a hierarchical context map for a codebase (agents.md + AGENTS.md)."""
    if model:
        discovery_model = generation_model = model
    base_url = base_url or os.getenv("OPENAI_BASE_URL")
    repo_root = root_dir.resolve()
    console.print(f"[bold]Scanning repository:[/bold] {repo_root}")

    client = _get_openai_client(base_url)
    selected_dirs = run_phase1_directory_selection(client, discovery_model, repo_root)
    console.print(f"[dim]Selected {len(selected_dirs)} directories for AGENTS.md[/dim]")

    agents_md_contents, folder_summaries = build_agents_md_contents(
        repo_root, selected_dirs, client, discovery_model, generation_model, templates_dir, max_lines
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
