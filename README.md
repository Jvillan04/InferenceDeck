# InferenceDeck

A terminal UI ([Textual](https://textual.textualize.io/) + Docker SDK) for running **vLLM** and **SGLang**
containers: find them, start/stop/restart them, tail their logs, see the compose file or `docker run`
behind them, check their GPU/RAM budget, and keep editable drafts of launch configs.

## Install

```bash
git clone https://github.com/Jvillan04/InferenceDeck.git
cd InferenceDeck
./install.sh            # venv + editable install in ~/.local/share/inferencedeck, launcher in ~/.local/bin
inferencedeck
```

`install.sh` checks for Python ≥ 3.10, git, the docker CLI/daemon access and `nvidia-smi`, and can be
re-run safely. Options: `--branch <name>`, `--dev` (test deps). Run outside a checkout, it clones into
`~/.local/share/inferencedeck/src` with your own git credentials.

## Update

Any of these does `git pull --ff-only` → `pip install -e` → drafts DB migrations:

- press **`U`** in the app (or *Check for updates* in the command palette, `ctrl+p`), then **Restart now**
- `inferencedeck update` (`--check` only reports)
- `./update.sh`

The app also checks quietly once a day at startup and shows a notification when commits are available
(set `INFERENCEDECK_NO_UPDATE_CHECK=1` to turn this off). Updates are refused if the install checkout has
local changes. Drafts survive updates: the schema is versioned and only ever migrated forward.

## Uninstall

```bash
inferencedeck uninstall            # remove the app, keep your drafts
inferencedeck uninstall --purge    # remove everything, including the drafts database
```

It lists what it will delete (with sizes) and asks before removing anything; add `--yes` to skip the
question. The same script is available as `./uninstall.sh` in the repo (and `./install.sh --uninstall`
still works). It removes the virtualenv, the cloned source and the `~/.local/bin/inferencedeck` link. A
checkout you installed from, your containers and your images are never touched.

## Main screen

A container is listed if its **name or image** contains `vllm` or `sglang` (case-insensitive). Docker is
polled every 2 s; the list is updated in place and the cursor stays on the same container (tracked by
ID) when containers come and go.

| Key | Action |
| --- | --- |
| `s` | Start (unpauses a paused container) |
| `t` | Stop (asks first) |
| `r` | Restart (asks first) |
| `Enter` / `l` | Tail logs of the highlighted container |
| `e` | Save the container's compose file or reconstructed `docker run` as a draft |
| `d` | Open drafts |
| `ctrl+r` / `F5` | Refresh now |
| `U` | Check for updates |
| `q` | Quit |

The list shows 🟢 running · 🟡 paused · 🔄 restarting · ⚪ created · 🔴 stopped, and `⏳ stopping…` etc.
right away while an action runs. Docker calls run in worker threads, so the UI never blocks, even during
a 30 s graceful stop.

Tabs:

- **Details**: image, state, ports, start time, health, restarts, exit code / OOM-killed, command.
- **Logs**: live follow of stdout/stderr. Switching containers closes the previous stream. After a
  start/restart the stream re-attaches on its own.
- **Resources**: engine settings parsed from the command line, container limits, host GPUs from
  `nvidia-smi` (▶ marks the GPUs this container gets, plus its own memory use), and checks (below).
- **Compose / Run**: if the container came from compose (`com.docker.compose.*` labels), the compose
  file(s) as they are on this host. Otherwise a best-effort **reconstructed `docker run`**.

## Resource checks

Read from the engine flags (`--gpu-memory-utilization`, `--mem-fraction-static`, `--tensor-parallel-size`/`--tp`,
`--pipeline-parallel-size`, `--dp-size`, `--max-model-len`/`--context-length`, `--swap-space`,
`--cpu-offload-gb`, …, including commands wrapped in `sh -c "…"`), from the container config (`--gpus` /
device requests / `NVIDIA_VISIBLE_DEVICES`, memory, CPUs, `--shm-size`, `--ipc`), and from `nvidia-smi`:

- ❌ tp × pp × dp needs more GPUs than the container is given
- ❌ fraction × GPU memory is more than what other processes leave free (a running container's own usage is subtracted)
- ❌ containers sharing a GPU whose fractions add up to more than 1.0
- ❌/⚠️ shared memory under 1 GiB without `--ipc=host` (an error with multi-GPU NCCL)
- ❌ RAM limit ≤ `swap-space + cpu-offload-gb`; ⚠️ RAM limit < 16 GiB; ⚠️ fewer than 2 CPUs
- ⚠️ fraction > 0.95; ❌ no GPUs requested at all

Without `nvidia-smi`, the config-only checks still run.

## Drafts

Drafts are `docker run` commands or compose files kept in SQLite at
`${XDG_DATA_HOME:-~/.local/share}/inferencedeck/inferencedeck.db`. Each save with changes adds a revision,
and each launch is logged with its exit code and output.

| Key | Action |
| --- | --- |
| `ctrl+n` | New draft from a template (vLLM/SGLang × run/compose, or empty) |
| `ctrl+s` | Save (adds a revision) |
| `ctrl+l` | Launch (saves first, shows the lint summary, asks to confirm) |
| `ctrl+r` | Revision history: diff against the editor (`d` toggles full text), `Enter` restores |
| `ctrl+t` | Duplicate (handy for trying variants) |
| `ctrl+g` | Set the working directory (compose project dir / cwd for relative paths) |
| `Delete` | Delete the draft (when the list has focus) |
| `F8` | Abort a running launch |
| `Esc` | Back (asks if there are unsaved edits) |

The editor re-lints as you type with the same checks as the Resources tab, and also flags a
`--name` that clashes with an existing container. **Launching** shells out, so commands run exactly as
written: run drafts go through `/bin/sh -c` (`\` continuations and `${VAR}` expansion work), and compose
drafts run `docker compose -f <draft> --project-directory <working dir> up -d`.

When a container is saved as a draft, secret-looking env vars (`*TOKEN*`, `*KEY*`, `*SECRET*`,
`*PASSWORD*`) are written as `"${NAME}"` references. Secrets stay out of the database and are filled in
from your shell's environment at launch.

## Development

```bash
./install.sh --dev      # or: pip install -e '.[dev]'
pytest -q
```

The tests don't need a Docker daemon or a GPU: they use a fake backend, canned `nvidia-smi` output, a
temporary SQLite DB and Textual's Pilot.

Layout: `backend.py` (Docker SDK), `resources.py` (flag/limit parsing, nvidia-smi, checks), `lint.py`,
`compose.py` (compose discovery, `docker run` reconstruction), `drafts.py` (SQLite store + migrations),
`launcher.py`, `updater.py`, `app.py` (main screen), `screens.py` (drafts editor and dialogs).

## License

[MIT](LICENSE)
