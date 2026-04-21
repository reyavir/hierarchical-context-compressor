# hierarchical-context-compressor

![hcc — hierarchical context maps for AI agents](docs/hcc_image.png)

CLI tool to generate AI-optimized hierarchical context maps for any codebase. Run it locally, or **do the same thing in GitHub Actions** as this repo: a workflow that checks out your code, installs `hcc`, runs it on `--root .`, and optionally commits the generated files (no separate Marketplace “action” required).

- **All repos**: root **agents.md** (table of contents) plus per-directory **AGENTS.md** operational manuals.
- **Web apps only**: also **llms.txt** at the root (same index as agents.md).

Generation uses a **three-phase** flow: (1) an LLM selects which directories get their own AGENTS.md from the full tree; (2) for each selected directory, discovery picks which files to read; (3) generation writes concise AGENTS.md content. Root agents.md is a **concise ToC + minimal overview** (guide, not encyclopedia); nested AGENTS.md are **concise** operational manuals (Setup & Commands, Code Style & Patterns, Implementation Details).

---

### Two ways to read this README

| You want… | Go to |
|-----------|--------|
| **Use `hcc` on your own repos** | [**Use hcc**](#use-hcc) — install from Git, then run on any folder. |
| **Change or test this project’s code** | [**Developing this repository**](#developing-this-repository) — clone, `uv sync`, run tests. |

---

## Use hcc

Install the `hcc` command from Git, set your API key, then point it at any project directory.

Requires **Python 3.10+** and [uv](https://docs.astral.sh/uv/).

1. **Install** (adds `hcc` to your PATH):

   ```bash
   uv tool install git+https://github.com/reyavir/hierarchical-context-compressor.git
   ```

2. **API key** — set `OPENAI_API_KEY` in your environment, or put it in a `.env` file in the **project you’re analyzing** (never commit secrets).

3. **Run** on the codebase you care about:

   ```bash
   hcc --root /path/to/your/repo
   ```

   That writes **agents.md**, per-directory **AGENTS.md**, and **llms.txt** when applicable. Preview without writing files:

   ```bash
   hcc --root /path/to/your/repo --dry-run
   ```

4. **Uninstall** (if needed):

   ```bash
   uv tool uninstall hierarchical-context-compressor
   ```

**Optional: custom prompts per folder type** — create e.g. `.hcc/templates/docs.md` (and `tests.md`, `core.md`, …) and run:

```bash
hcc --root /path/to/repo --templates-dir .hcc/templates
```

---

## CLI options

| Option | Default | Description |
|--------|---------|-------------|
| `--root` | `.` | Repository root to analyze. |
| `--discovery-model` | `gpt-4o-mini` | Model for directory selection and file discovery. |
| `--generation-model` | `gpt-4o` | Model for AGENTS.md generation. |
| `--model` | — | Use this model for both discovery and generation (overrides the two above). |
| `--base-url` | — | Custom API base URL (e.g. LiteLLM proxy). |
| `--dry-run` | `false` | Do not write files; print a tree view. |
| `--templates-dir` | — | Directory with optional per-type templates: `docs.md`, `tests.md`, `core.md`, `infra.md`, `generic.md`. If present, the file content is used as the system prompt for that directory type. |
| `--max-dirs` | `15` | Maximum number of directories selected in phase 1 for AGENTS.md generation. Increase for large repos with many important code areas. |
| `--changed-only` | `false` | Regenerate AGENTS only for directories touched by git changes (plus their ancestors and root). Useful for faster incremental CI runs. |
| `--changed-base` | `HEAD~1` | Git base ref used by `--changed-only` for diff range: `<base>...HEAD`. |
| `--selection-report` | — | Write JSON diagnostics for directory selection (before/after filtering counts and directory lists). Helpful for CI debugging and tuning. |

**Output size:** Each generated `AGENTS.md` **body** (everything below the `### Local Agent Context` header) is limited to **100 lines**; if the model returns more, the rest is cut and a short truncation note is appended.

### Common workflows

```bash
# Full generation
hcc --root .

# Preview only (no writes)
hcc --root . --dry-run

# Incremental generation from git changes
hcc --root . --changed-only --changed-base origin/main

# Incremental generation + diagnostics report
hcc --root . --changed-only --changed-base origin/main --selection-report .hcc/selection-report.json
```

When `--changed-only` is enabled, hcc computes changes with git diff range `<base>...HEAD`, keeps only selected directories that are ancestors of changed files, and always retains root (`.`).

Example `--selection-report` payload:

```json
{
  "changedOnly": true,
  "changedBase": "origin/main",
  "counts": {
    "selectedBefore": 12,
    "selectedAfter": 4
  },
  "selectedDirectoriesAfter": [".", "src", "src/api", "tests"]
}
```

**Environment**

- `OPENAI_API_KEY` – required for LLM generation.
- `OPENAI_BASE_URL` – optional; used when `--base-url` is not set (e.g. `OPENAI_BASE_URL=http://localhost:4000` for a proxy).

Example with a proxy:

```bash
hcc --base-url http://localhost:4000 --root /path/to/repo
```

---

## GitHub Actions (your repo)

You can run `hcc` in CI the same way **this** project does: add a workflow under `.github/workflows/` that installs the tool, runs `hcc --root .` (or `python -m src.main --root .`), and—if you want—commits `agents.md`, nested `AGENTS.md`, and `llms.txt` when they change.

- Add repo secret **`OPENAI_API_KEY`**.
- If the workflow should **push** commits, set **Actions → Workflow permissions → Read and write** for the repository.

**Reference:** [`.github/workflows/generate-context.yml`](.github/workflows/generate-context.yml) is what runs on pushes to `main` *here* (`pip install -e .` from the checkout). For **your** repo, the usual approach is `pip install "git+https://github.com/reyavir/hierarchical-context-compressor.git"` then `hcc --root .`—see **[`.github/workflows/hcc-template-for-other-repos.yml.example`](.github/workflows/hcc-template-for-other-repos.yml.example)** to copy and adapt.

---

## Developing this repository

Use this section only if you are **working on the hierarchical-context-compressor source code** (not required to run `hcc` on other projects).

```bash
git clone https://github.com/reyavir/hierarchical-context-compressor.git
cd hierarchical-context-compressor
uv sync
# or: pip install -e .
```

Run the CLI from this clone:

```bash
uv run hcc --root /path/to/some/repo
# or
uv run python -m src.main --root /path/to/some/repo
```

Run tests:

```bash
uv run pytest
```

**Traditional pip** (editable install from a clone only):

```bash
pip install -e .
python -m src.main --root /path/to/your/repo
```
