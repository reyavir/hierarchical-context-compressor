"""Tests for formatter with selected_dirs + repo_root."""
from __future__ import annotations

from pathlib import Path

from src.formatter import format_root_agents_md, format_root_llms, wrap_agents_md_header


def test_format_root_agents_md_with_selected_dirs(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    sub = repo / "src"
    sub.mkdir()
    selected = [repo, sub]
    summaries = {repo: "Root summary.", sub: "Source code."}
    out = format_root_agents_md(repo, selected, summaries)
    assert "Repository index" in out
    assert "table of contents" in out.lower()
    assert "**.**" in out or ".**" in out
    assert "AGENTS.md" in out
    assert "Root summary" in out
    assert "Source code" in out


def test_format_root_llms_with_selected_dirs(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    sub = repo / "docs"
    sub.mkdir()
    selected = [repo, sub]
    summaries = {repo: "Root", sub: "Docs"}
    out = format_root_llms(repo, selected, summaries)
    assert "Master Dispatcher" in out
    assert "AGENTS.md" in out


def test_wrap_agents_md_header(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    sub = repo / "src"
    sub.mkdir()
    body = "## Setup & Commands\n\nDone."
    out = wrap_agents_md_header(sub, repo, body)
    assert "Local Agent Context" in out
    assert "Scope" in out
    assert "Setup & Commands" in out
    assert "src" in out
