# hierarchical-context-compressor

CLI and GitHub Action to generate AI-optimized hierarchical context maps for any codebase. Produces a root **llms.txt** (Master Dispatcher) and per-directory **agents.md** (Local Agent Context) files so AI agents can navigate your repo like a sitemap.

Define structure once; get purpose, public API, and dependencies per folder — with optional LLM summaries when `OPENAI_API_KEY` is set.

## Installation

Requires **Python 3.10+** and [uv](https://docs.astral.sh/uv/).

```bash
uv tool install git+https://github.com/YOUR_ORG/hierarchical-context-compressor.git
```

This installs the **`hcc`** command globally.

To uninstall:

```bash
uv tool uninstall hierarchical-context-compressor
```

## Development

To work on the tool itself, clone the repo and sync dependencies:

```bash
git clone https://github.com/YOUR_ORG/hierarchical-context-compressor.git
cd hierarchical-context-compressor
uv sync
```

Run the CLI from the repo:

```bash
uv run hcc --root /path/to/some/repo
# or
uv run python -m src.main --root /path/to/some/repo
```

Run tests:

```bash
uv run pytest
```

## Quick Start

1. **Generate context for a repo** (writes `llms.txt` and `agents.md` into the target repo):

   ```bash
   hcc --root /path/to/your/repo
   ```

2. **Preview without writing files** (dry run with a tree and summaries):

   ```bash
   hcc --root /path/to/your/repo --dry-run
   ```

3. **Optional: LLM summaries**  
   Set `OPENAI_API_KEY` in your environment, or when developing from a clone add a `.env` file in the project root (see `.gitignore`; never commit it).

## Traditional pip

```bash
pip install -r requirements.txt
python -m src.main --root /path/to/your/repo
```

Replace `YOUR_ORG` in the git URLs with your GitHub org or username once the repo is pushed.
