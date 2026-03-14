from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Set

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

SOURCE_EXTENSIONS: Set[str] = {
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".java",
    ".go",
    ".rs",
    ".rb",
}


@dataclass
class FolderInfo:
    path: Path
    rel_path: Path
    files: List[Path] = field(default_factory=list)
    subfolders: List["FolderInfo"] = field(default_factory=list)

    @property
    def source_files(self) -> List[Path]:
        return [f for f in self.files if f.suffix in SOURCE_EXTENSIONS]

    @property
    def has_readme(self) -> bool:
        return any(f.name.lower() == "readme.md" for f in self.files)

    @property
    def has_agents_md(self) -> bool:
        return any(f.name.lower() == "agents.md" for f in self.files)


def load_gitignore(root: Path) -> PathSpec | None:
    gitignore = root / ".gitignore"
    if not gitignore.exists():
        return None

    patterns: Iterable[str] = gitignore.read_text().splitlines()
    return PathSpec.from_lines("gitwildmatch", patterns)


def is_ignored(path: Path, root: Path, spec: PathSpec | None) -> bool:
    if spec is None:
        # Always ignore common junk directories even without .gitignore
        parts = set(path.parts)
        return any(d in parts for d in DEFAULT_IGNORE_DIRS)

    rel = path.relative_to(root).as_posix()
    if spec.match_file(rel):
        return True

    # Also apply default dir ignores on top of .gitignore
    parts = set(path.parts)
    return any(d in parts for d in DEFAULT_IGNORE_DIRS)


def crawl_repository(root: Path) -> FolderInfo:
    """
    Walk the repository from root, respecting .gitignore and default junk directories.
    Returns a tree of FolderInfo objects rooted at `root`.
    """
    root = root.resolve()
    spec = load_gitignore(root)

    def build(folder_path: Path) -> FolderInfo:
        rel = folder_path.relative_to(root)
        folder = FolderInfo(path=folder_path, rel_path=rel)

        entries = sorted(folder_path.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        for entry in entries:
            if is_ignored(entry, root, spec):
                continue
            if entry.is_dir():
                sub = build(entry)
                # Skip empty folders entirely
                if sub.files or sub.subfolders:
                    folder.subfolders.append(sub)
            elif entry.is_file():
                folder.files.append(entry)
        return folder

    return build(root)

