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

You MAY use these level-2 headers when (and only when) you have concrete, evidence-backed content:

## Setup & Commands
- List ONLY commands you can point to in real files (e.g. package.json scripts, Makefile, pyproject.toml). For each: give the **exact command** and **source** (e.g. `npm run test` from `package.json` scripts). If you find no runnable commands, omit this section entirely. Do not guess or invent commands.

## Code Style & Patterns
- Mention ONLY patterns you can back with a concrete example: file name, function/class name, or config key. Include at least one concrete path or symbol per bullet (e.g. `src/api/user.py:get_user`, `tests/test_user.py`). If you do not see any clear style signals, omit this section.

## Implementation Details
- Name **specific entrypoints and modules** (e.g. `main.py`, `app.py`, `src/index.tsx`) and state which file to open first for a given kind of change. Focus on how code is wired: imports, module boundaries, layers. Do not restate the README. If there is nothing interesting beyond what the tree already shows, omit this section.

**Banned:** Do not use generic phrases ("well structured", "best practices", "clean code") without immediately following with a concrete example (file path or symbol in backticks). If a bullet has no code/path reference, omit it.

Use at most the three headers above. If you have no strong signal for a section, leave it out entirely. Be concise; bullets over paragraphs."""

# --- Directory type classification and type-specific prompts ---
# Type -> (display name, persona name, persona description for header)
DIRECTORY_TYPE_META: Dict[str, tuple[str, str, str]] = {
    "core": ("Core", "Senior Lead Engineer", "Architectural rules, exported APIs, and strict logic boundaries."),
    "docs": ("Docs", "Technical Writer", "Tone of voice, formatting rules (Markdown/Docusaurus), and link-checking commands."),
    "tests": ("Tests", "QA Engineer", "Test coverage requirements, mock patterns, and how to run specific suites."),
    "infra": ("Infra", "DevOps Specialist", "Environment variables, deployment safety, and CLI flag explanations."),
    "generic": ("General", "Agent", "No specific role; extract setup, style, and implementation details from the tree and files."),
}

# Type -> full system prompt for generation (persona + boundary + what to extract)
TYPE_SYSTEM_PROMPTS: Dict[str, str] = {
    "core": """You are writing an Operational Manual for a **Senior Lead Engineer** in this directory (core/source code). Focus on: architectural rules, exported APIs, and strict logic boundaries. Extract USEFUL, SPECIFIC information only; every bullet must reference a concrete file path, command, or symbol (in backticks).

**Boundary:** This agent owns production code. Preserve exported APIs and architectural contracts; do not suggest breaking changes without calling out impact.

You MAY use these level-2 headers only when you have evidence-backed content:
## Setup & Commands — Exact build/run commands from package.json, Makefile, pyproject.toml (e.g. `npm run build`, `uv run dev`). Omit if none.
## Code Style & Patterns — Concrete patterns with file/symbol references (e.g. `src/api/user.py:get_user`). Omit if no clear signals.
## Implementation Details — Entrypoints (`main.py`, `app.py`), module boundaries, key imports. Omit if nothing beyond the tree.

Leave out any section that has no unique signal. No generic phrases without a concrete example.""",

    "docs": """You are writing an Operational Manual for a **Technical Writer** in this directory (documentation). Focus on: tone of voice, formatting rules (Markdown/Docusaurus/etc.), and link-checking or lint-docs commands. Extract USEFUL, SPECIFIC information only; every bullet must reference a concrete path or command (in backticks).

**Boundary:** Only write in Markdown; do not suggest code changes to application source. Stay within docs tooling and structure.

You MAY use these level-2 headers only when you have evidence-backed content:
## Setup & Commands — How to build/serve the docs site, lint-docs, link-check (e.g. `npm run docs:build`, `mkdocs serve`). Omit if none.
## Code Style & Patterns — Doc formatting rules, sidebar structure, cross-linking conventions. Omit if no clear signals.
## Implementation Details — Docs framework (Docusaurus, MkDocs), config file location, asset layout. Omit if nothing specific.

Leave out any section that has no unique signal. No generic phrases without a concrete example.""",

    "tests": """You are writing an Operational Manual for a **QA Engineer** in this directory (tests). Focus on: test framework used, how to run tests, and mocking strategy. Extract USEFUL, SPECIFIC information only; every bullet must reference a concrete path or command (in backticks).

**Boundary:** Never modify source code to make tests pass. Only change test code, fixtures, or config. Preserve intended behavior.

You MAY use these level-2 headers only when you have evidence-backed content:
## Setup & Commands — How to run tests (e.g. `pytest tests/`, `npm run test`, `jest --config`). Include filter flags for specific suites. Omit if none.
## Code Style & Patterns — Test naming, mock patterns (e.g. `tests/conftest.py`, `jest.mock`). Omit if no clear signals.
## Implementation Details — Test layout, fixtures, coverage config. Omit if nothing specific.

Leave out any section that has no unique signal. No generic phrases without a concrete example.""",

    "infra": """You are writing an Operational Manual for a **DevOps Specialist** in this directory (scripts/infra). Focus on: environment variables, deployment safety, and CLI flag explanations. Extract USEFUL, SPECIFIC information only; every bullet must reference a concrete path or command (in backticks).

**Boundary:** Do not change application code. Only modify scripts, Docker, CI, or Terraform. Call out destructive or irreversible actions.

You MAY use these level-2 headers only when you have evidence-backed content:
## Setup & Commands — How to run scripts, deploy, or validate (e.g. `docker build`, `terraform plan`). Include required env vars. Omit if none.
## Code Style & Patterns — Script conventions, config file locations. Omit if no clear signals.
## Implementation Details — Pipeline stages, secrets handling, rollback steps. Omit if nothing specific.

Leave out any section that has no unique signal. No generic phrases without a concrete example.""",

    "generic": """You are writing an Operational Manual for an AI agent in this directory. No specific persona—extract whatever is useful from the file tree and contents. Every bullet must reference a concrete file path, command, or symbol (in backticks). No broad or vague statements.

You MAY use these level-2 headers only when you have evidence-backed content:
## Setup & Commands — Runnable commands from config files (package.json, Makefile, etc.). Omit if none.
## Code Style & Patterns — Concrete patterns with file/symbol references. Omit if no clear signals.
## Implementation Details — Entrypoints, key modules, how things are wired. Omit if nothing specific.

Leave out any section that has no unique signal. No generic phrases without a concrete example.""",
}


def _get_system_prompt_for_type(type_key: str, templates_dir: Optional[Path] = None) -> str:
    """
    Return the system prompt for a directory type. If templates_dir is set and contains
    {type_key}.md or {type_key}.txt, use that file's content; otherwise use TYPE_SYSTEM_PROMPTS.
    """
    if templates_dir is not None and templates_dir.is_dir():
        for ext in (".md", ".txt"):
            path = templates_dir / f"{type_key}{ext}"
            if path.is_file():
                return path.read_text(encoding="utf-8", errors="replace").strip()
    return TYPE_SYSTEM_PROMPTS.get(type_key, AGENTS_MD_SYSTEM_PROMPT)


def _classify_directory(dir_path: Path, repo_root: Path) -> tuple[str, str, str]:
    """
    Classify directory into type based on path and contents. Returns (type_key, persona_name, persona_description).
    Order: infra -> docs -> tests -> core (default).
    """
    try:
        rel_parts = dir_path.resolve().relative_to(repo_root.resolve()).parts
    except ValueError:
        rel_parts = ()
    path_lower = "/".join(rel_parts).lower()
    try:
        names = {p.name.lower() for p in dir_path.iterdir()}
        files = [p.name.lower() for p in dir_path.iterdir() if p.is_file()]
    except OSError:
        names = set()
        files = []

    # INFRA: path has scripts, infra, terraform; or has Dockerfile, .yml
    if any(p in path_lower for p in ("scripts", "infra", "terraform", ".github")):
        meta = DIRECTORY_TYPE_META["infra"]
        return ("infra", meta[1], meta[2])
    if "dockerfile" in names or any(f.endswith(".yml") or f.endswith(".yaml") for f in files):
        meta = DIRECTORY_TYPE_META["infra"]
        return ("infra", meta[1], meta[2])

    # DOCS: path has docs; or directory is mostly .md files
    if "docs" in path_lower:
        meta = DIRECTORY_TYPE_META["docs"]
        return ("docs", meta[1], meta[2])
    md_count = sum(1 for f in files if f.endswith(".md"))
    if md_count >= 2 and len(files) <= 10:
        meta = DIRECTORY_TYPE_META["docs"]
        return ("docs", meta[1], meta[2])

    # TESTS: path has tests, test, spec; or files like test_*.py, *.spec.*
    if any(p in path_lower for p in ("tests", "test", "spec", "__tests__")):
        meta = DIRECTORY_TYPE_META["tests"]
        return ("tests", meta[1], meta[2])
    if any(f.startswith("test_") or ".spec." in f for f in files):
        meta = DIRECTORY_TYPE_META["tests"]
        return ("tests", meta[1], meta[2])

    # CORE: only if strong signal—path is a known source root or dir has multiple source files
    source_roots = ("src", "app", "lib", "core", "packages", "pkg")
    if any(p in path_lower for p in source_roots):
        meta = DIRECTORY_TYPE_META["core"]
        return ("core", meta[1], meta[2])
    source_extensions = (".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".rs", ".java")
    source_count = sum(1 for f in files if any(f.endswith(ext) for ext in source_extensions))
    if source_count >= 2:
        meta = DIRECTORY_TYPE_META["core"]
        return ("core", meta[1], meta[2])

    # GENERIC: fallback when nothing clearly fits (misc folders, config-only, single file, etc.)
    meta = DIRECTORY_TYPE_META["generic"]
    return ("generic", meta[1], meta[2])


# Caps
MAX_DIRS_PHASE1 = 15
MAX_PATHS_DISCOVERY = 20
MAX_CHARS_PER_FILE = 8000
MAX_TOTAL_CHARS_DISCOVERY = 40000


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


def generate_agents_md_with_llm(
    client: Optional[OpenAI],
    generation_model: str,
    dir_path: Path,
    repo_root: Path,
    tree_text: str,
    discovered_contents: Dict[str, str],
    is_root: bool,
    templates_dir: Optional[Path] = None,
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
        type_key, _, _ = _classify_directory(dir_path, repo_root)
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
@click.option(
    "--templates-dir",
    "templates_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Directory with optional per-type templates: docs.md, tests.md, core.md, infra.md, generic.md. If present, used as the system prompt for that directory type.",
)
def cli(
    root_dir: Path,
    discovery_model: str,
    generation_model: str,
    model: Optional[str],
    base_url: Optional[str],
    dry_run: bool,
    templates_dir: Optional[Path],
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
        repo_root, selected_dirs, client, discovery_model, generation_model, templates_dir
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
