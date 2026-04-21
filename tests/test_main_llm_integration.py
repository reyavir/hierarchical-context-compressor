"""Tests for main three-phase flow, parsing, and write_context_files."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from click.testing import CliRunner

import src.main as main_mod
from src.main import (
    cli,
    _parse_directory_selection,
    _parse_discovery_paths,
    _classify_directory,
    _get_system_prompt_for_type,
    _extract_summary,
    _prune_sections,
    _limit_agents_md_body_lines,
    generate_agents_md_with_llm,
    build_agents_md_contents,
    write_context_files,
    _merge_with_existing_agents_md,
    run_phase2_discovery,
    get_changed_files_since,
    filter_selected_dirs_for_changed_files,
)


class FakeChoiceMessage(SimpleNamespace):
    content: str


class FakeChoice(SimpleNamespace):
    message: FakeChoiceMessage


def test_parse_directory_selection_includes_root(tmp_path: Path) -> None:
    out = _parse_directory_selection(".", tmp_path)
    assert out == [tmp_path]
    out = _parse_directory_selection(".\nsrc\ndocs", tmp_path)
    (tmp_path / "src").mkdir()
    (tmp_path / "docs").mkdir()
    out = _parse_directory_selection(".\nsrc\ndocs", tmp_path)
    assert tmp_path in out
    assert (tmp_path / "src") in out
    assert (tmp_path / "docs") in out


def test_parse_directory_selection_skips_invalid(tmp_path: Path) -> None:
    (tmp_path / "real").mkdir()
    out = _parse_directory_selection(".\nreal\n../etc\nnonexistent", tmp_path)
    assert tmp_path in out
    assert (tmp_path / "real") in out
    assert len(out) == 2


def test_classify_directory(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "src").mkdir()
    (repo / "docs").mkdir()
    (repo / "tests").mkdir()
    (repo / "scripts").mkdir()
    assert _classify_directory(repo / "docs", repo) == "docs"
    assert _classify_directory(repo / "tests", repo) == "tests"
    assert _classify_directory(repo / "scripts", repo) == "infra"
    assert _classify_directory(repo / "src", repo) == "core"
    (repo / "misc").mkdir()
    (repo / "misc" / "readme.txt").write_text("hi", encoding="utf-8")
    assert _classify_directory(repo / "misc", repo) == "generic"


def test_parse_discovery_paths(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("", encoding="utf-8")
    (tmp_path / "b.txt").write_text("", encoding="utf-8")
    out = _parse_discovery_paths("a.py\nb.txt\nc.missing", tmp_path)
    assert "a.py" in out
    assert "b.txt" in out
    assert "c.missing" not in out


def test_generate_agents_md_with_llm_nested(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    sub = repo / "src"
    sub.mkdir()
    (sub / "main.py").write_text("def main(): pass", encoding="utf-8")
    tree = ".\n└── main.py"
    contents = {"main.py": "def main(): pass"}

    class FakeCompletions:
        def create(self, model, messages, max_tokens):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content="## Setup & Commands\n\n`pip install .`\n\n## Code Style & Patterns\n\nPython.\n\n## Implementation Details\n\nEntry: main.py."
                        )
                    )
                ]
            )

    client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    out = generate_agents_md_with_llm(
        client, "gpt-4o", sub, repo, tree, contents, is_root=False
    )
    assert "## Setup & Commands" in out
    assert "Local Agent Context" in out


def test_get_system_prompt_for_type_uses_template_when_provided(tmp_path: Path) -> None:
    """When --templates-dir contains docs.md (or docs.txt), that content is used for docs type."""
    templates_dir = tmp_path / "templates"
    templates_dir.mkdir()
    (templates_dir / "docs.md").write_text("You are a custom docs template. Focus on xyz.", encoding="utf-8")
    out = _get_system_prompt_for_type("docs", templates_dir)
    assert "custom docs template" in out
    assert "xyz" in out
    # Other types still use built-in when no template file
    out_core = _get_system_prompt_for_type("core", templates_dir)
    assert "Senior Lead Engineer" in out_core


def test_get_system_prompt_for_type_fallback_without_templates_dir() -> None:
    """Without templates_dir, built-in prompt is used."""
    out = _get_system_prompt_for_type("docs", None)
    assert "Technical Writer" in out
    assert "Operational Manual" in out


def test_generate_agents_md_with_llm_no_client(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    out = generate_agents_md_with_llm(
        None, "gpt-4o", repo, repo, ".", {}, is_root=True
    )
    assert "Local Agent Context" in out
    assert "Table of contents" in out or "table of contents" in out


def test_build_agents_md_contents_no_client_uses_root_only(tmp_path: Path, monkeypatch) -> None:
    import src.main as main_mod
    monkeypatch.setattr(main_mod, "_get_openai_client", lambda base_url=None: None)
    monkeypatch.setattr(
        main_mod,
        "run_phase1_directory_selection",
        lambda client, model, root: [root],
    )
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "README.md").write_text("# Hi", encoding="utf-8")
    selected = [repo]
    contents, summaries = build_agents_md_contents(
        repo, selected, None, "gpt-4o-mini", "gpt-4o"
    )
    assert repo in contents
    assert "Local Agent Context" in contents[repo]
    assert repo in summaries


def test_write_context_files_emits_agents_md(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "README.md").write_text("# Root", encoding="utf-8")
    selected = [repo]
    contents = {repo: "### Local Agent Context: Repository Root\n\n## Scope\n\n...\n\n## Table of contents\n\n- [x](./x)"}
    summaries = {repo: "Root summary"}
    write_context_files(repo, selected, contents, summaries)
    assert (repo / "agents.md").exists()
    assert "table of contents" in (repo / "agents.md").read_text(encoding="utf-8").lower() or "Repository index" in (repo / "agents.md").read_text(encoding="utf-8")
    assert (repo / "AGENTS.md").exists()
    assert "Local Agent Context" in (repo / "AGENTS.md").read_text(encoding="utf-8")


def test_write_context_files_backend_only_no_llms_txt(tmp_path: Path) -> None:
    """Backend-only repo (e.g. FastAPI API) does not get llms.txt."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text(
        '[project]\nname = "api"\ndependencies = ["fastapi"]',
        encoding="utf-8",
    )
    (repo / "main.py").write_text("from fastapi import FastAPI\napp = FastAPI()", encoding="utf-8")
    selected = [repo]
    contents = {repo: "### Local Agent Context\n\n## Scope\n\n..."}
    summaries = {repo: "Root"}
    write_context_files(repo, selected, contents, summaries)
    assert (repo / "agents.md").exists()
    assert not (repo / "llms.txt").exists()


def test_write_context_files_web_app_emits_llms_txt(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "package.json").write_text(
        '{"scripts":{"start":"next start"},"dependencies":{"next":"14.0.0"}}',
        encoding="utf-8",
    )
    selected = [repo]
    contents = {repo: "### Local Agent Context\n\n## Scope\n\n..."}
    summaries = {repo: "Root"}
    write_context_files(repo, selected, contents, summaries)
    assert (repo / "llms.txt").exists()
    assert "Master Dispatcher" in (repo / "llms.txt").read_text(encoding="utf-8")


def test_merge_with_existing_agents_md_preserves_rules(tmp_path: Path) -> None:
    folder = tmp_path / "repo"
    folder.mkdir()
    agents_path = folder / "AGENTS.md"
    agents_path.write_text(
        "### Local Agent Context: X\n\n## Scope\n\n...\n\n## Rules\n\n- Keep me\n",
        encoding="utf-8",
    )
    regenerated = "### Local Agent Context: X\n\n## Scope\n\nNew\n\n## Setup & Commands\n\nFresh\n"
    merged = _merge_with_existing_agents_md(agents_path, regenerated)
    assert "## Setup & Commands" in merged
    assert "- Keep me" in merged


def test_extract_summary_prefers_substantive_line() -> None:
    content = (
        "### Local Agent Context: X\n\n"
        "## Setup & Commands\n\n"
        "- `pip install .`\n"
    )
    result = _extract_summary(content)
    assert "pip" in result or "`pip" in result or "- " in result
    assert result != "Setup & Commands"


def test_extract_summary_fallback() -> None:
    content = "### Local\n\n## A\n\n## B"
    result = _extract_summary(content)
    assert result.strip() in ("A", "B", "Local")


def test_prune_sections_drops_weak_section() -> None:
    md = (
        "## First\n\n"
        "One line only.\n\n"
        "## Second\n\n"
        "Body with `code` here.\n"
    )
    result = _prune_sections(md)
    assert "## First" not in result
    assert "One line only" not in result
    assert "## Second" in result
    assert "`code`" in result


def test_limit_agents_md_body_lines_noop_when_short() -> None:
    s = "a\nb\nc"
    assert _limit_agents_md_body_lines(s, max_lines=100) == s


def test_limit_agents_md_body_lines_truncates() -> None:
    long = "\n".join([f"line {i}" for i in range(150)])
    out = _limit_agents_md_body_lines(long, max_lines=100)
    assert "truncated" in out
    assert "line 99" in out
    assert "line 100" not in out.split("_…")[0]


def test_get_system_prompt_for_type_empty_template_uses_builtin(tmp_path: Path) -> None:
    templates_dir = tmp_path / "templates"
    templates_dir.mkdir()
    (templates_dir / "docs.md").write_text("", encoding="utf-8")
    out = _get_system_prompt_for_type("docs", templates_dir)
    assert "Technical Writer" in out
    assert out.strip() != ""


def test_merge_with_existing_agents_md_preserves_rules_case_insensitive(tmp_path: Path) -> None:
    folder = tmp_path / "repo"
    folder.mkdir()
    agents_path = folder / "AGENTS.md"
    agents_path.write_text(
        "### Local Agent Context: X\n\n## Scope\n\n...\n\n## rules\n\n- Keep me\n",
        encoding="utf-8",
    )
    regenerated = "### Local Agent Context: X\n\n## Scope\n\nNew\n\n## Setup & Commands\n\nFresh\n"
    merged = _merge_with_existing_agents_md(agents_path, regenerated)
    assert "- Keep me" in merged


def test_run_phase2_discovery_no_client_returns_fallback(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# Hello", encoding="utf-8")
    result = run_phase2_discovery(None, "gpt-4o-mini", tmp_path)
    assert isinstance(result, dict)
    assert "README.md" in result


def test_filter_selected_dirs_for_changed_files_keeps_ancestors(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    src = repo / "src"
    src.mkdir()
    docs = repo / "docs"
    docs.mkdir()
    api = src / "api"
    api.mkdir()
    selected = [repo, src, docs, api]
    changed_files = ["src/api/handler.py"]
    out = filter_selected_dirs_for_changed_files(selected, changed_files, repo)
    assert out == [repo, src, api]


def test_filter_selected_dirs_for_changed_files_no_matches_keeps_root(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    src = repo / "src"
    src.mkdir()
    selected = [repo, src]
    out = filter_selected_dirs_for_changed_files(selected, ["README.md"], repo)
    assert out == [repo]


def test_get_changed_files_since_parses_git_output(tmp_path: Path, monkeypatch) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    class FakeCompleted:
        def __init__(self):
            self.stdout = "src/main.py\ndocs/readme.md\n"

    def fake_run(*args, **kwargs):
        return FakeCompleted()

    monkeypatch.setattr(main_mod.subprocess, "run", fake_run)
    out = get_changed_files_since(repo, "HEAD~1")
    assert out == ["src/main.py", "docs/readme.md"]


def test_cli_changed_only_filters_selected_dirs(monkeypatch, tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    src = repo / "src"
    src.mkdir()
    docs = repo / "docs"
    docs.mkdir()

    captured = {}
    monkeypatch.setattr(main_mod, "_get_openai_client", lambda base_url=None: None)
    monkeypatch.setattr(main_mod, "run_phase1_directory_selection", lambda *args, **kwargs: [repo, src, docs])
    monkeypatch.setattr(main_mod, "get_changed_files_since", lambda *args, **kwargs: ["src/main.py"])
    monkeypatch.setattr(main_mod, "render_tree", lambda *args, **kwargs: None)

    def fake_build(*args, **kwargs):
        selected_dirs = args[1]
        captured["selected"] = selected_dirs
        return ({repo: "### Local Agent Context\n\n## Table of contents\n\n- x"}, {repo: "Root"})

    monkeypatch.setattr(main_mod, "build_agents_md_contents", fake_build)
    runner = CliRunner()
    result = runner.invoke(cli, ["--root", str(repo), "--dry-run", "--changed-only"])
    assert result.exit_code == 0
    assert captured["selected"] == [repo, src]
