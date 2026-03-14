from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Set, Tuple

from .crawler import FolderInfo, SOURCE_EXTENSIONS


@dataclass
class FolderAnalysis:
    folder: FolderInfo
    is_high_signal: bool
    key_exports: List[str]
    dependencies: Set[str]
    main_file: Path | None
    main_file_head: str | None


def is_high_signal_folder(folder: FolderInfo) -> bool:
    """
    Signal test:
    - High signal if folder has > 3 source files OR contains a README.md.
    """
    return len(folder.source_files) > 3 or folder.has_readme


def choose_main_file(folder: FolderInfo) -> Path | None:
    """
    Heuristic: pick the "main" file in a folder for LLM summarization.
    Priority:
    - file named like 'main', 'app', 'index' (any extension)
    - otherwise, the first source file alphabetically.
    """
    if not folder.source_files:
        return None

    candidates = {p.name.lower(): p for p in folder.source_files}
    for stem in ("main", "app", "index"):
        for name, path in candidates.items():
            if name.startswith(stem + ".") or name.endswith("/" + stem):
                return path

    return sorted(folder.source_files, key=lambda p: p.name.lower())[0]


def read_head(path: Path, max_lines: int = 20) -> str:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            lines = []
            for i, line in enumerate(f):
                if i >= max_lines:
                    break
                lines.append(line.rstrip("\n"))
        return "\n".join(lines)
    except OSError:
        return ""


def extract_key_symbols_and_deps(folder: FolderInfo) -> Tuple[List[str], Set[str]]:
    """
    Very lightweight static analysis:
    - For Python / JS / TS files, look for top-level function and class declarations.
    - Collect import targets as dependencies.
    """
    key_symbols: List[str] = []
    deps: Set[str] = set()

    for source in folder.source_files:
        text = ""
        try:
            text = source.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        ext = source.suffix
        if ext == ".py":
            _py_symbols_and_deps(text, key_symbols, deps)
        elif ext in {".js", ".jsx", ".ts", ".tsx"}:
            _js_ts_symbols_and_deps(text, key_symbols, deps)
        else:
            # For other languages, just record the file stem as a "module".
            deps.add(source.stem)

    # Deduplicate while preserving order
    seen: Set[str] = set()
    unique_symbols: List[str] = []
    for s in key_symbols:
        if s not in seen:
            seen.add(s)
            unique_symbols.append(s)

    return unique_symbols, deps


def _py_symbols_and_deps(text: str, key_symbols: List[str], deps: Set[str]) -> None:
    import re

    func_re = re.compile(r"^def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", re.MULTILINE)
    class_re = re.compile(r"^class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*[:\(]", re.MULTILINE)
    import_re = re.compile(
        r"^(?:from\s+([a-zA-Z0-9_\.]+)\s+import|import\s+([a-zA-Z0-9_\.]+))", re.MULTILINE
    )

    for m in func_re.finditer(text):
        name = m.group(1)
        if not name.startswith("_"):
            key_symbols.append(name)

    for m in class_re.finditer(text):
        name = m.group(1)
        if not name.startswith("_"):
            key_symbols.append(name)

    for m in import_re.finditer(text):
        mod = m.group(1) or m.group(2)
        if mod:
            deps.add(mod.split(".")[0])


def _js_ts_symbols_and_deps(text: str, key_symbols: List[str], deps: Set[str]) -> None:
    import re

    func_re = re.compile(
        r"^(?:export\s+)?function\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(",
        re.MULTILINE,
    )
    class_re = re.compile(
        r"^(?:export\s+)?class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*[{]",
        re.MULTILINE,
    )
    import_re = re.compile(
        r"""^(?:import\s+(?:[^'"]+\s+from\s+)?|require\()\s*['"]([^'"]+)['"]""",
        re.MULTILINE,
    )

    for m in func_re.finditer(text):
        key_symbols.append(m.group(1))

    for m in class_re.finditer(text):
        key_symbols.append(m.group(1))

    for m in import_re.finditer(text):
        spec = m.group(1)
        if not spec.startswith("."):
            deps.add(spec.split("/")[0])


def analyze_folder(folder: FolderInfo) -> FolderAnalysis:
    is_high = is_high_signal_folder(folder)
    key_exports, deps = extract_key_symbols_and_deps(folder)
    main_file = choose_main_file(folder)
    main_head = read_head(main_file) if main_file else None
    return FolderAnalysis(
        folder=folder,
        is_high_signal=is_high,
        key_exports=key_exports,
        dependencies=deps,
        main_file=main_file,
        main_file_head=main_head,
    )

