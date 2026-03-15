"""
Folder tree utilities: recursive tree as text, list dirs, read files by path.
Respects .gitignore and default ignore dirs. No FolderInfo/FolderAnalysis.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Set

from pathspec import PathSpec


DEFAULT_IGNORE_DIRS: Set[str] = {
    ".git",
    ".hg",
    ".svn",
    ".idea",
    ".vscode",
    "node_modules",
    ".venv",
    "venv",
    "__pycache__",
    "dist",
    "build",
    ".pytest_cache",
}


def load_gitignore(root: Path) -> PathSpec | None:
    gitignore = root / ".gitignore"
    if not gitignore.exists():
        return None
    patterns: Iterable[str] = gitignore.read_text().splitlines()
    return PathSpec.from_lines("gitwildmatch", patterns)


def is_ignored(path: Path, root: Path, spec: PathSpec | None) -> bool:
    if spec is None:
        parts = set(path.parts)
        return any(d in parts for d in DEFAULT_IGNORE_DIRS)
    rel = path.relative_to(root).as_posix()
    if spec.match_file(rel):
        return True
    parts = set(path.parts)
    return any(d in parts for d in DEFAULT_IGNORE_DIRS)


def get_tree(root: Path, prefix: str = "") -> str:
    """
    Recursive file tree as text for the given path. Respects .gitignore.
    Format: unicode tree like "├── file" and "└── dir/"
    """
    root = root.resolve()
    spec = load_gitignore(root)
    lines: List[str] = []

    def walk(folder_path: Path, p: str, is_root: bool) -> None:
        if is_root:
            lines.append(".")
        entries: List[tuple[str, bool, Path | None]] = []
        try:
            children = sorted(folder_path.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
        except OSError:
            return
        for entry in children:
            if is_ignored(entry, root, spec):
                continue
            if entry.is_file():
                entries.append((entry.name, False, None))
            elif entry.is_dir():
                entries.append((entry.name + "/", True, entry))
        for i, (name, is_dir, sub_path) in enumerate(entries):
            is_last = i == len(entries) - 1
            branch = "└── " if is_last else "├── "
            lines.append(p + branch + name)
            if is_dir and sub_path is not None:
                ext = "    " if is_last else "│   "
                walk(sub_path, p + ext, False)

    walk(root, prefix, prefix == "")
    return "\n".join(lines)


def list_directories(root: Path) -> List[Path]:
    """List direct subdirectory paths under root (respecting .gitignore)."""
    root = root.resolve()
    spec = load_gitignore(root)
    result: List[Path] = []
    try:
        for entry in sorted(root.iterdir(), key=lambda p: p.name.lower()):
            if entry.is_dir() and not is_ignored(entry, root, spec):
                result.append(entry)
    except OSError:
        pass
    return result


def read_files_by_paths(
    root: Path,
    relative_paths: List[str],
    max_chars_per_file: int = 8000,
    max_total_chars: int = 40000,
) -> Dict[str, str]:
    """
    Read files by relative paths under root. Skip paths outside root (security).
    Truncate per file and enforce total cap. Return dict path -> content.
    """
    root = root.resolve()
    result: Dict[str, str] = {}
    total = 0

    for rel in relative_paths:
        if total >= max_total_chars:
            break
        rel = rel.strip().lstrip("/")
        if not rel:
            continue
        path = (root / rel).resolve()
        try:
            if not path.is_file():
                continue
            path.relative_to(root)
        except (ValueError, OSError):
            continue
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        chunk = raw[:max_chars_per_file] if len(raw) > max_chars_per_file else raw
        if len(raw) > max_chars_per_file:
            chunk += "\n... (truncated)"
        result[rel] = chunk
        total += len(chunk)

    return result
