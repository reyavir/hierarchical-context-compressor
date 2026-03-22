# hierarchical-context-compressor

CLI tool to generate AI-optimized hierarchical context maps for any codebase. Run it locally, or **do the same thing in GitHub Actions** as this repo: a workflow that checks out your code, installs `hcc`, runs it on `--root .`, and optionally commits the generated files (no separate Marketplace “action” required).

- **All repos**: root **agents.md** (table of contents) plus per-directory **AGENTS.md** operational manuals.
- **Web apps only**: also **llms.txt** at the root (same index as agents.md).

Generation uses a **three-phase** flow: (1) an LLM selects which directories get their own AGENTS.md from the full tree; (2) for each selected directory, discovery picks which files to read; (3) generation writes concise AGENTS.md content. Root agents.md is a **concise ToC + minimal overview** (guide, not encyclopedia); nested AGENTS.md are **concise** operational manuals (Setup & Commands, Code Style & Patterns, Implementation Details).

## Installation

Requires **Python 3.10+** and [uv](https://docs.astral.sh/uv/).

```bash
uv tool install git+https://github.com/reyavir/hierarchical-context-compressor.git
```

This installs the **`hcc`** command globally.

To uninstall:

```bash
uv tool uninstall hierarchical-context-compressor
```

## Development

Clone the repo and install in editable mode:

```bash
git clone https://github.com/reyavir/hierarchical-context-compressor.git
cd hierarchical-context-compressor
uv sync
# or: pip install -e .
```

Run the CLI:

```bash
uv run hcc --root /path/to/some/repo
# or
uv run python -m src.main --root /path/to/some/repo
```

Run tests:

```bash
uv run pytest
```

## CLI options

| Option | Default | Description |
|--------|---------|-------------|
| `--root` | `.` | Repository root to analyze. |
| `--discovery-model` | `gpt-4o-mini` | Model for directory selection and file discovery. |
| `--generation-model` | `gpt-4o` | Model for AGENTS.md generation. |
| `--model` | — | Use this model for both discovery and generation (overrides the two above). |
| `--base-url` | — | Custom API base URL (e.g. LiteLLM proxy). |
| `--dry-run` | `false` | Do not write files; print a tree view. |
| `--templates-dir` | — | Directory with optional per-type templates: `docs.md`, `tests.md`, `core.md`, `infra.md`, `generic.md`. If present, the file content is used as the system prompt for that directory type (e.g. for docs folders use your own “xyz” template). |

**Environment**

- `OPENAI_API_KEY` – required for LLM generation.
- `OPENAI_BASE_URL` – optional; used when `--base-url` is not set (e.g. `OPENAI_BASE_URL=http://localhost:4000` for a proxy).

Example with a proxy:

```bash
hcc --base-url http://localhost:4000 --root /path/to/repo
# or
OPENAI_BASE_URL=http://localhost:4000 hcc --root /path/to/repo
```

## Quick start

1. **Generate context** (writes root **agents.md** and per-directory **AGENTS.md**; web apps also get **llms.txt**):

   ```bash
   hcc --root /path/to/your/repo
   ```

2. **Preview without writing** (dry run):

   ```bash
   hcc --root /path/to/your/repo --dry-run
   ```

3. **Optional: Custom templates per type**  
   For a given directory type (e.g. docs), you can supply your own system prompt. Create a directory (e.g. `.hcc/templates`) and add files named by type: `docs.md`, `tests.md`, `core.md`, `infra.md`, or `generic.md`. The content of that file is used as the LLM system prompt when generating AGENTS.md for that type. Example: for docs folders, put your “xyz” template in `.hcc/templates/docs.md` and run:

   ```bash
   hcc --root /path/to/repo --templates-dir .hcc/templates
   ```

4. **Optional: LLM generation**  
   Set `OPENAI_API_KEY` in your environment, or add a `.env` file in the project root (see `.gitignore`; never commit it).

## Traditional pip

```bash
pip install -e .
python -m src.main --root /path/to/your/repo
```

## GitHub Actions (your repo)

You can run `hcc` in CI the same way **this** project does: add a workflow under `.github/workflows/` that installs the tool, runs `hcc --root .` (or `python -m src.main --root .`), and—if you want—commits `agents.md`, nested `AGENTS.md`, and `llms.txt` when they change.

- Add repo secret **`OPENAI_API_KEY`**.
- If the workflow should **push** commits, set **Actions → Workflow permissions → Read and write** for the repository.

**Reference:** [`.github/workflows/generate-context.yml`](.github/workflows/generate-context.yml) is what runs on pushes to `main` *here* (`pip install -e .` from the checkout). For **your** repo, the usual approach is `pip install "git+https://github.com/reyavir/hierarchical-context-compressor.git"` then `hcc --root .`—see **[`.github/workflows/hcc-template-for-other-repos.yml.example`](.github/workflows/hcc-template-for-other-repos.yml.example)** to copy and adapt.
