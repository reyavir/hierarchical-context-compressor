"""
Format root agents.md and llms.txt from selected_dirs and repo_root.
No FolderInfo/FolderAnalysis.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List


def _display_name(repo_root: Path, dir_path: Path) -> str:
    if dir_path.resolve() == repo_root.resolve():
        return "."
    try:
        return dir_path.relative_to(repo_root).as_posix()
    except ValueError:
        return dir_path.as_posix()


def format_root_agents_md(
    repo_root: Path,
    selected_dirs: List[Path],
    folder_summaries: Dict[Path, str],
) -> str:
    """
    Build the root agents.md table of contents. Links to AGENTS.md for each selected dir.
    """
    lines: List[str] = []
    lines.append("# Repository index (agents.md)")
    lines.append("")
    lines.append(
        "This file is the **table of contents** for this repository. Each link points to an **AGENTS.md** "
        "for that directory. Use it to navigate to the right directory before running tasks or editing code."
    )
    lines.append("")
    lines.append("## Table of contents")
    lines.append("")
    for dir_path in selected_dirs:
        display = _display_name(repo_root, dir_path)
        rel = dir_path.relative_to(repo_root).as_posix() if dir_path != repo_root else "."
        link_path = f"./{rel}/AGENTS.md" if rel != "." else "./AGENTS.md"
        context_link = f" ([AGENTS.md]({link_path}))"
        summary = folder_summaries.get(dir_path, "").strip()
        if summary:
            lines.append(f"- **{display}**{context_link}: {summary}")
        else:
            lines.append(f"- **{display}**{context_link}")
    lines.append("")
    return "\n".join(lines)


def format_root_llms(
    repo_root: Path,
    selected_dirs: List[Path],
    folder_summaries: Dict[Path, str],
) -> str:
    """
    Build the top-level llms.txt for web apps. Master index linking to AGENTS.md per directory.
    """
    lines: List[str] = []
    lines.append("# Master Dispatcher (llms.txt)")
    lines.append("")
    lines.append(
        "This file is the **web app** master index. Each link points to an **AGENTS.md** for that directory."
    )
    lines.append("")
    lines.append("## Table of contents")
    lines.append("")
    for dir_path in selected_dirs:
        display = _display_name(repo_root, dir_path)
        rel = dir_path.relative_to(repo_root).as_posix() if dir_path != repo_root else "."
        link_path = f"./{rel}/AGENTS.md" if rel != "." else "./AGENTS.md"
        context_link = f" ([AGENTS.md]({link_path}))"
        summary = folder_summaries.get(dir_path, "").strip()
        if summary:
            lines.append(f"- **{display}**{context_link}: {summary}")
        else:
            lines.append(f"- **{display}**{context_link}")
    lines.append("")
    return "\n".join(lines)


def wrap_agents_md_header(dir_path: Path, repo_root: Path, body: str) -> str:
    """
    Prepend the standard Local Agent Context header and Scope to the LLM-generated AGENTS.md body.
    """
    display = _display_name(repo_root, dir_path)
    folder_name = "Repository Root" if display == "." else dir_path.name
    header = f"### Local Agent Context: {folder_name}\n\n"
    return header + body.strip() + "\n"
