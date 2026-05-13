# Core Utilities

Shared project helpers live here.

Current contents:

- `config.py` for loading `configs/default.yaml`
- `resolve_project_path()` for turning repo-relative paths into absolute paths

This package stays intentionally small so the rest of the repo can depend on one clear config entrypoint.
