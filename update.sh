#!/usr/bin/env bash
# Update an InferenceDeck installed with install.sh: git pull --ff-only, reinstall, migrate drafts DB.
#
#   ./update.sh           update now
#   ./update.sh --check   only report whether an update is available
#
# Same code path as `inferencedeck update` and the U key inside the app.
set -euo pipefail

PREFIX="${INFERENCEDECK_HOME:-${XDG_DATA_HOME:-$HOME/.local/share}/inferencedeck}"
PY="$PREFIX/venv/bin/python"

if [ ! -x "$PY" ]; then
    printf '\033[1;31merror:\033[0m InferenceDeck is not installed in %s; run install.sh first.\n' "$PREFIX" >&2
    exit 1
fi
exec "$PY" -m inferencedeck update "$@"
