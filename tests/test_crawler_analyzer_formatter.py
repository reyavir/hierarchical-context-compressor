from __future__ import annotations

import os
from pathlib import Path

from src.crawler import FolderInfo, crawl_repository, SOURCE_EXTENSIONS
from src.analyzer import (
    analyze_folder,
    is_high_signal_folder,
    extract_key_symbols_and_deps,
    FolderAnalysis,
)
from src.formatter import format_root_llms, format_agents_md


def make_fake_folder(tmp_path: Path) -> FolderInfo:
    """
    Helper to construct a small in-memory folder tree on disk and then crawl it.
    """
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.py").write_text("import os\n\n\ndef foo():\n    return 1\n", encoding="utf-8")
    (root / "b.py").write_text("class Bar:\n    pass\n", encoding="utf-8")
    (root / "README.md").write_text("# Readme", encoding="utf-8")

    sub = root / "sub"
    sub.mkdir()
    (sub / "main.py").write_text("def main():\n    pass\n", encoding="utf-8")
    (sub / "util.js").write_text("export function helper() {}", encoding="utf-8")

    return crawl_repository(root)


def test_crawl_repository_respects_source_extensions(tmp_path: Path) -> None:
    folder_tree = make_fake_folder(tmp_path)
    root = folder_tree

    assert root.source_files  # a.py and b.py
    assert all(p.suffix in SOURCE_EXTENSIONS for p in root.source_files)

    # Ensure subfolder got captured
    assert any(sub.rel_path.as_posix() == "sub" for sub in root.subfolders)


def test_is_high_signal_folder_conditions(tmp_path: Path) -> None:
    folder_tree = make_fake_folder(tmp_path)
    root = folder_tree

    # Root has README.md and multiple source files -> high signal
    assert is_high_signal_folder(root) is True

    # Subfolder has 2 source files and no README -> not high signal by rule (>3 sources OR README)
    sub = next(sub for sub in root.subfolders if sub.rel_path.as_posix() == "sub")
    assert len(sub.source_files) == 2
    assert sub.has_readme is False
    assert is_high_signal_folder(sub) is False


def test_analyze_folder_extracts_symbols_and_deps(tmp_path: Path) -> None:
    folder_tree = make_fake_folder(tmp_path)
    root = folder_tree

    analysis = analyze_folder(root)
    assert isinstance(analysis, FolderAnalysis)

    # From a.py and b.py
    assert "foo" in analysis.key_exports
    assert "Bar" in analysis.key_exports
    assert "os" in analysis.dependencies


def test_format_folder_context_structure(tmp_path: Path) -> None:
    folder_tree = make_fake_folder(tmp_path)
    root = folder_tree
    analysis = analyze_folder(root)

    content = format_agents_md(analysis, folder_summary="**Root folder** summary.")

    assert "### Local Agent Context" in content
    assert "## Scope" in content
    assert "This agent is responsible for the logic within this directory and its subdirectories." in content
    assert "## System Context" in content
    assert "### Folder Purpose" in content
    assert "### Public API" in content
    assert "### Dependencies" in content
    assert "### Files" in content
    # No raw code should be present (only names and filenames)
    assert "def foo" not in content
    assert "class Bar" not in content


def test_format_root_llms_includes_local_map_links(tmp_path: Path) -> None:
    folder_tree = make_fake_folder(tmp_path)
    root = folder_tree

    # Mark root as high-signal, sub as not
    root_analysis = analyze_folder(root)
    sub = next(sub for sub in root.subfolders if sub.rel_path.as_posix() == "sub")
    sub_analysis = analyze_folder(sub)

    analyses = {
        root.path: root_analysis,
        sub.path: sub_analysis,
    }
    summaries = {
        root.path: "Root summary.",
        sub.path: "Sub summary.",
    }

    llms = format_root_llms(root, analyses, summaries)

    # Root is high-signal because of README
    assert "([agents](./agents.md))" in llms
    # Sub is not high-signal, so no link
    assert "sub/agents.md" not in llms

