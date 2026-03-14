from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from .analyzer import FolderAnalysis
from .crawler import FolderInfo


def _folder_display_name(folder: FolderInfo) -> str:
    return "." if folder.rel_path == Path(".") else folder.rel_path.as_posix()


def format_root_llms(
    root: FolderInfo,
    analyses: Dict[Path, FolderAnalysis],
    folder_summaries: Dict[Path, str],
) -> str:
    """
    Build the top-level llms.txt "sitemap" file.
    """
    lines: List[str] = []
    lines.append("# Master Dispatcher (llms.txt)")
    lines.append("")
    lines.append(
        "This file is the top-level routing map for AI agents. Follow links to per-folder `agents.md` files "
        "to get local responsibilities, public APIs, and dependencies."
    )
    lines.append("")
    lines.append("## Conventions")
    lines.append("- **llms.txt**: master dispatcher for the repo.")
    lines.append("- **agents.md**: local agent context for a directory (and its subdirectories).")
    lines.append("")
    lines.append("## Modules")
    lines.append("")

    def walk(folder: FolderInfo, indent: int = 0) -> None:
        analysis = analyses.get(folder.path)
        summary = folder_summaries.get(folder.path)
        bullet = "  " * indent + "- "

        display_name = _folder_display_name(folder)

        context_link = ""
        if analysis and (analysis.is_high_signal or folder.rel_path == Path(".")):
            # Link to local agents.md; path is relative to repo root.
            ctx_rel = folder.rel_path / "agents.md"
            link_path = f"./{ctx_rel.as_posix()}" if ctx_rel.as_posix() != "agents.md" else "./agents.md"
            context_link = f" ([agents]({link_path}))"

        desc = summary.strip() if summary else ""
        if desc:
            lines.append(f"{bullet}**{display_name}**{context_link}: {desc}")
        else:
            lines.append(f"{bullet}**{display_name}**{context_link}")

        for sub in folder.subfolders:
            walk(sub, indent + 1)

    walk(root, indent=0)
    lines.append("")
    return "\n".join(lines)


def format_agents_md(
    analysis: FolderAnalysis,
    folder_summary: Optional[str],
    public_api: Optional[Sequence[str]] = None,
) -> str:
    """
    Build the contents of an `agents.md` for a single directory.
    """
    folder = analysis.folder
    rel = _folder_display_name(folder)
    folder_name = folder.path.name if rel != "." else "Repository Root"

    lines: List[str] = []
    lines.append(f"### Local Agent Context: {folder_name}")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("This agent is responsible for the logic within this directory and its subdirectories.")
    lines.append("")

    lines.append("## System Context")
    lines.append("")

    # Purpose
    purpose = folder_summary.strip() if folder_summary else ""
    if not purpose:
        purpose = f"This folder `{rel}` contains source files used by the application."
    lines.append("### Folder Purpose")
    lines.append("")
    lines.append(purpose)
    lines.append("")

    # Public API (LLM-identified preferred, fallback to static extraction)
    lines.append("### Public API")
    lines.append("")
    api_list = list(public_api) if public_api else []
    if not api_list:
        api_list = list(analysis.key_exports)
    if api_list:
        for name in api_list:
            lines.append(f"- `{name}`")
    else:
        lines.append("_No public API symbols were detected._")
    lines.append("")

    # Dependencies
    lines.append("### Dependencies")
    lines.append("")
    if analysis.dependencies:
        for dep in sorted(analysis.dependencies):
            lines.append(f"- `{dep}`")
    else:
        lines.append("_No external dependencies detected beyond this folder._")
    lines.append("")

    # Files overview for extra structure, without raw code.
    lines.append("### Files")
    lines.append("")
    if folder.files:
        for f in sorted(folder.files, key=lambda p: p.name.lower()):
            rel_file = f.relative_to(folder.path).as_posix()
            lines.append(f"- `{rel_file}`")
    else:
        lines.append("_This folder currently has no files._")
    lines.append("")

    return "\n".join(lines)

