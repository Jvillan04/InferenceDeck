# InferenceDeck

Python/Textual TUI for managing vLLM and SGLang containers. Main screen: `inferencedeck/app.py`;
drafts editor and dialogs: `inferencedeck/screens.py`. See README.md for features and layout.

## Branches

- Name every branch after the feature it implements, in kebab-case: `gpu-metrics-panel`,
  `drafts-export`, `fix-log-reconnect`. Never use auto-generated or random names. If a session hands
  you one (e.g. `claude/<adjective>-<name>-<id>`), create a feature-named branch from it and push only that.
- Branch from the latest `main`, one feature per branch.

## Checks before pushing or merging

```bash
pytest -q                          # no Docker daemon or GPU needed
shellcheck install.sh update.sh
```
