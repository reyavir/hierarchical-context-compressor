"""Tests for src.tree module."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.tree import get_tree, list_directories, read_files_by_paths


def test_get_tree_basic(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x", encoding="utf-8")
    (tmp_path / "b.txt").write_text("y", encoding="utf-8")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "c.js").write_text("z", encoding="utf-8")
    tree = get_tree(tmp_path)
    assert "." in tree
    assert "a.py" in tree
    assert "b.txt" in tree
    assert "sub" in tree or "sub/" in tree
    assert "c.js" in tree
    assert "├──" in tree or "└──" in tree


def test_get_tree_respects_gitignore(tmp_path: Path) -> None:
    (tmp_path / "visible.py").write_text("", encoding="utf-8")
    (tmp_path / ".gitignore").write_text("visible.py\n", encoding="utf-8")
    tree = get_tree(tmp_path)
    assert "visible.py" not in tree


def test_list_directories(tmp_path: Path) -> None:
    (tmp_path / "file.txt").write_text("", encoding="utf-8")
    d1 = tmp_path / "dir1"
    d1.mkdir()
    d2 = tmp_path / "dir2"
    d2.mkdir()
    dirs = list_directories(tmp_path)
    assert len(dirs) == 2
    assert d1 in dirs
    assert d2 in dirs


def test_read_files_by_paths(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
    (tmp_path / "b.txt").write_text("world", encoding="utf-8")
    result = read_files_by_paths(tmp_path, ["a.txt", "b.txt"])
    assert result["a.txt"] == "hello"
    assert result["b.txt"] == "world"


def test_read_files_by_paths_outside_root_rejected(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("ok", encoding="utf-8")
    other = tmp_path / "other"
    other.mkdir()
    (other / "b.txt").write_text("nope", encoding="utf-8")
    result = read_files_by_paths(tmp_path, ["a.txt", "other/b.txt"])
    assert "a.txt" in result
    result = read_files_by_paths(tmp_path, ["../other/b.txt"])
    assert len(result) == 0


def test_read_files_by_paths_truncates(tmp_path: Path) -> None:
    (tmp_path / "big.txt").write_text("x" * 20000, encoding="utf-8")
    result = read_files_by_paths(
        tmp_path, ["big.txt"], max_chars_per_file=1000, max_total_chars=5000
    )
    assert len(result["big.txt"]) <= 1000 + len("\n... (truncated)")
    assert "... (truncated)" in result["big.txt"]


def test_read_files_by_paths_skips_nonexistent(tmp_path: Path) -> None:
    (tmp_path / "exists.txt").write_text("yes", encoding="utf-8")
    result = read_files_by_paths(tmp_path, ["exists.txt", "missing.txt", ""])
    assert result == {"exists.txt": "yes"}
