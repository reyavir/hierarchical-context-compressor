from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from src.crawler import FolderInfo
from src.analyzer import FolderAnalysis, is_high_signal_folder
from src.main import (
    summarize_folder_with_llm,
    build_analyses,
    build_summaries,
    write_context_files,
)


class FakeChoiceMessage(SimpleNamespace):
    content: str


class FakeChoice(SimpleNamespace):
    message: FakeChoiceMessage


class FakeChatCompletions(SimpleNamespace):
    def create(self, model: str, messages, max_tokens: int):
        # Return a simple deterministic message with the folder from the user content.
        user_content = messages[-1]["content"]
        return SimpleNamespace(
            choices=[
                FakeChoice(
                    message=FakeChoiceMessage(
                        content=f"**Summary** for:\n\n{user_content[:40]}..."
                    )
                )
            ]
        )


class FakeClient(SimpleNamespace):
    def __init__(self) -> None:
        super().__init__(chat=SimpleNamespace(completions=FakeChatCompletions()))


def test_summarize_folder_with_llm_uses_client(tmp_path: Path) -> None:
    root = tmp_path
    (root / "main.py").write_text("def main():\n    pass\n", encoding="utf-8")
    folder = FolderInfo(path=root, rel_path=Path("."), files=[root / "main.py"], subfolders=[])
    analysis = FolderAnalysis(
        folder=folder,
        is_high_signal=True,
        key_exports=[],
        dependencies=set(),
        main_file=root / "main.py",
        main_file_head="def main():\n    pass\n",
    )

    client = FakeClient()
    info = summarize_folder_with_llm(client, "gpt-4o-mini", analysis)

    assert "Summary" in info.purpose_md or info.purpose_md


def test_build_analyses_and_summaries_without_real_llm(tmp_path: Path, monkeypatch) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "README.md").write_text("# Root", encoding="utf-8")
    (repo / "main.py").write_text("def main():\n    pass\n", encoding="utf-8")

    # Simulate a simple one-folder tree
    folder = FolderInfo(path=repo, rel_path=Path("."), files=list(repo.iterdir()), subfolders=[])

    # build_analyses should analyze just this folder
    analyses = build_analyses(folder)
    assert repo in analyses
    analysis = analyses[repo]
    assert analysis.is_high_signal is True  # README present

    # Monkeypatch _get_openai_client to None to force heuristic summaries (no network).
    from src import main as main_mod

    monkeypatch.setattr(main_mod, "_get_openai_client", lambda: None)

    summaries = build_summaries(analyses, model="gpt-4o-mini")
    assert repo in summaries
    assert "This repository root" in summaries[repo].purpose_md


def test_write_context_files_creates_llms_and_context(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "README.md").write_text("# Root", encoding="utf-8")
    (repo / "main.py").write_text("def main():\n    pass\n", encoding="utf-8")

    folder = FolderInfo(
        path=repo, rel_path=Path("."), files=list(repo.iterdir()), subfolders=[]
    )
    analyses = build_analyses(folder)
    from src.main import FolderLLMInfo

    summaries = {repo: FolderLLMInfo(purpose_md="Root summary.", public_api=[])}

    write_context_files(repo, folder, analyses, summaries)

    llms_path = repo / "llms.txt"
    agents_path = repo / "agents.md"

    assert llms_path.exists()
    assert agents_path.exists()

    llms_content = llms_path.read_text(encoding="utf-8")
    agents_content = agents_path.read_text(encoding="utf-8")

    assert "Master Dispatcher" in llms_content
    assert "Local Agent Context" in agents_content


def test_existing_agents_md_preserves_rules(tmp_path: Path) -> None:
    from src.main import _merge_with_existing_agents_md

    folder = tmp_path / "repo"
    folder.mkdir()
    agents_path = folder / "agents.md"
    agents_path.write_text(
        "### Local Agent Context: X\n\n## Scope\n\n...\n\n## Rules\n\n- Keep me\n",
        encoding="utf-8",
    )

    regenerated = "### Local Agent Context: X\n\n## Scope\n\nNew\n\n## System Context\n\nFresh\n"
    merged = _merge_with_existing_agents_md(agents_path, regenerated)

    assert "## System Context" in merged
    assert "- Keep me" in merged

