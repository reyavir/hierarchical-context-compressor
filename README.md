# hierarchical-context-compressor

CLI tool and GitHub Action to generate a hierarchical, agent-friendly context map for a codebase:

- `llms.txt` as the **Master Dispatcher** at the root.
- Per-directory `agents.md` files as **Local Agent Contexts** with scope, system context, public API, dependencies, and files.

Usage (example):

```bash
cd /path/to/hierarchical-context-compressor
pip install -r requirements.txt
python -m src.main --root /path/to/your/repo
```

