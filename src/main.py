from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

import click
from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.tree import Tree

from .analyzer import FolderAnalysis, analyze_folder
from .crawler import FolderInfo, crawl_repository
from .formatter import format_agents_md, format_root_llms


console = Console()


SYSTEM_PROMPT = (
    "You are a technical documentarian. Summarize this folder's purpose in 2 sentences. "
    "Focus on 'What it does' and 'How it relates to the rest of the app'. "
    "Do not use fluff. Output in Markdown format."
)

PUBLIC_API_INSTRUCTIONS = (
    "Also identify the folder's Public API: names of functions/classes that other folders should call/use. "
    "List names only, as Markdown bullets using backticks.\n\n"
    "Output format:\n"
    "## Purpose\n"
    "<2 sentences>\n\n"
    "## Public API\n"
    "- `Name`\n"
)


@dataclass(frozen=True)
class FolderLLMInfo:
    purpose_md: str
    public_api: Sequence[str]


def _get_openai_client() -> Optional[OpenAI]:
    # Load .env from project root (two levels up from this file: src/ -> project/)
    project_root = Path(__file__).resolve().parents[1]
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    # The OpenAI client auto-reads the API key from env.
    return OpenAI()


def summarize_folder_with_llm(
    client: Optional[OpenAI],
    model: str,
    analysis: FolderAnalysis,
) -> FolderLLMInfo:
    """
    Use a small LLM call to summarize a folder's purpose based on file names
    and the head of the main file.
    Falls back to a heuristic summary if no client is available.
    """
    folder = analysis.folder
    rel = folder.rel_path.as_posix()

    file_list = "\n".join(f"- {p.name}" for p in folder.files)
    snippet = analysis.main_file_head or ""

    user_content = f"""Folder: `{rel}`

Files:
{file_list or '(no files)'}

Main file snippet:
```
{snippet}
```"""

    if client is None:
        # Heuristic fallback: derive a more concrete description from static analysis.
        purpose = _heuristic_purpose(analysis)
        return FolderLLMInfo(purpose_md=purpose, public_api=list(analysis.key_exports))

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT + "\n\n" + PUBLIC_API_INSTRUCTIONS},
                {"role": "user", "content": user_content},
            ],
            max_tokens=160,
        )
        content = resp.choices[0].message.content or ""
        purpose, public_api = _parse_llm_folder_output(content.strip())
        if not public_api:
            public_api = list(analysis.key_exports)
        if not purpose:
            purpose = content.strip()
        return FolderLLMInfo(purpose_md=purpose, public_api=public_api)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[yellow]LLM summarization failed for {rel}: {exc}[/yellow]")
        purpose = _heuristic_purpose(analysis)
        return FolderLLMInfo(purpose_md=purpose, public_api=list(analysis.key_exports))


def _parse_llm_folder_output(md: str) -> tuple[str, list[str]]:
    purpose = ""
    purpose_match = re.search(
        r"^##\s+Purpose\s*\n([\s\S]*?)(?:\n##\s+|\Z)", md, re.MULTILINE
    )
    if purpose_match:
        purpose = purpose_match.group(1).strip()

    api: list[str] = []
    api_match = re.search(
        r"^##\s+Public API\s*\n([\s\S]*?)(?:\n##\s+|\Z)", md, re.MULTILINE
    )
    if api_match:
        api_block = api_match.group(1)
        for line in api_block.splitlines():
            m = re.search(r"`([^`]+)`", line)
            if m:
                api.append(m.group(1).strip())

    seen = set()
    uniq: list[str] = []
    for name in api:
        if name and name not in seen:
            seen.add(name)
            uniq.append(name)
    return purpose, uniq


def _heuristic_purpose(analysis: FolderAnalysis) -> str:
    """
    Token-efficient but more informative fallback description when no LLM is available.
    """
    folder = analysis.folder
    rel = folder.rel_path.as_posix() or "."

    exports = analysis.key_exports
    deps = sorted(analysis.dependencies)
    subdirs = [sub.path.name for sub in folder.subfolders]

    parts = []

    if rel == ".":
        parts.append("This repository root contains the main application entrypoint and core modules.")
    else:
        parts.append(f"This folder `{rel}` contains logic focused on a specific feature or layer of the app.")

    if exports:
        shown = ", ".join(f"`{name}`" for name in exports[:6])
        if len(exports) > 6:
            shown += ", ..."
        parts.append(f"It exposes a public surface via symbols such as {shown}.")

    if deps:
        shown = ", ".join(f"`{name}`" for name in deps[:6])
        parts.append(f"It depends on external modules like {shown}.")

    if subdirs:
        subs = ", ".join(f"`{name}`" for name in subdirs[:4])
        parts.append(f"It is organized into subdirectories {subs} for more focused responsibilities.")

    if len(parts) == 1:
        # Ensure at least two sentences per original instruction.
        parts.append("It participates in the overall application by collaborating with neighboring folders and modules.")

    return " ".join(parts)


def build_analyses(root_folder: FolderInfo) -> Dict[Path, FolderAnalysis]:
    analyses: Dict[Path, FolderAnalysis] = {}

    def walk(folder: FolderInfo) -> None:
        analyses[folder.path] = analyze_folder(folder)
        for sub in folder.subfolders:
            walk(sub)

    walk(root_folder)
    return analyses


def build_summaries(
    analyses: Dict[Path, FolderAnalysis],
    model: str,
) -> Dict[Path, FolderLLMInfo]:
    client = _get_openai_client()
    summaries: Dict[Path, FolderLLMInfo] = {}

    for path, analysis in analyses.items():
        # We always summarize high-signal folders, and the root folder.
        if not analysis.is_high_signal and analysis.folder.rel_path != Path("."):
            continue
        summaries[path] = summarize_folder_with_llm(client, model, analysis)

    return summaries


def render_tree(
    root: FolderInfo,
    analyses: Dict[Path, FolderAnalysis],
    summaries: Dict[Path, FolderLLMInfo],
) -> None:
    """
    Render a rich tree view for dry-run mode.
    """
    def node_for(folder: FolderInfo) -> Tree:
        analysis = analyses[folder.path]
        label = folder.rel_path.as_posix() or "."
        meta = []
        if analysis.is_high_signal:
            meta.append("high-signal")
        if folder.has_readme:
            meta.append("readme")
        meta_str = f" [{' ,'.join(meta)}]" if meta else ""
        title = f"[bold]{label}[/bold]{meta_str}"
        tree = Tree(title)

        # Files
        for f in sorted(folder.files, key=lambda p: p.name.lower()):
            tree.add(f"[cyan]{f.name}[/cyan]")

        # Children
        for sub in folder.subfolders:
            tree.add(node_for(sub))
        return tree

    root_tree = node_for(root)
    console.print(root_tree)

    console.print()
    console.print("[bold magenta]Summaries[/bold magenta]")
    for path, info in summaries.items():
        rel = analyses[path].folder.rel_path.as_posix() or "."
        console.print(f"[bold]{rel}[/bold]: {info.purpose_md}")
        if info.public_api:
            console.print("[dim]Public API:[/dim] " + ", ".join(info.public_api))
        console.print()


def write_context_files(
    repo_root: Path,
    root_folder: FolderInfo,
    analyses: Dict[Path, FolderAnalysis],
    summaries: Dict[Path, FolderLLMInfo],
) -> None:
    llms_path = repo_root / "llms.txt"
    llms_path.write_text(
        format_root_llms(
            root_folder,
            analyses,
            {k: v.purpose_md for k, v in summaries.items()},
        ),
        encoding="utf-8",
    )

    # Per high-signal folder agents.md files + always root agents.md
    for path, analysis in analyses.items():
        if not (analysis.is_high_signal or analysis.folder.rel_path == Path(".")):
            continue
        info = summaries.get(path)
        purpose = info.purpose_md if info else None
        public_api = info.public_api if info else None
        content = format_agents_md(analysis, purpose, public_api=public_api)
        agents_path = path / "agents.md"
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
    base = regenerated.rstrip()
    return base + "\n\n" + rules_block + "\n"


@click.command()
@click.option(
    "--root",
    "root_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("."),
    help="Root directory of the repository to analyze.",
)
@click.option(
    "--model",
    type=str,
    default="gpt-4o-mini",
    show_default=True,
    help="OpenAI model to use for folder summarization.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Do not write files; instead print a rich tree view of the computed context.",
)
def cli(root_dir: Path, model: str, dry_run: bool) -> None:
    """
    Generate a hierarchical context map for a codebase.
    """
    repo_root = root_dir.resolve()
    console.print(f"[bold]Scanning repository:[/bold] {repo_root}")

    root_folder = crawl_repository(repo_root)
    analyses = build_analyses(root_folder)
    summaries = build_summaries(analyses, model=model)

    if dry_run:
        console.print("[bold blue]Dry run mode - not writing any files.[/bold blue]")
        render_tree(root_folder, analyses, summaries)
    else:
        console.print("[bold green]Writing llms.txt and agents.md files...[/bold green]")
        write_context_files(repo_root, root_folder, analyses, summaries)
        console.print("[bold green]Done.[/bold green]")


if __name__ == "__main__":
    cli()

