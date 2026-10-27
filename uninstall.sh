#!/usr/bin/env bash
# Remove an InferenceDeck install made by install.sh.
#
#   ./uninstall.sh            remove the app, keep your drafts database
#   ./uninstall.sh --purge    also delete the drafts database (everything under INFERENCEDECK_HOME)
#   ./uninstall.sh --yes      don't ask for confirmation
#
# Also available as `inferencedeck uninstall [--purge] [--yes]` and `install.sh --uninstall [--purge]`.
# Never touches a checkout you installed from, your containers, or your images.
set -euo pipefail

PREFIX="${INFERENCEDECK_HOME:-${XDG_DATA_HOME:-$HOME/.local/share}/inferencedeck}"
BIN_DIR="${INFERENCEDECK_BIN:-$HOME/.local/bin}"
VENV="$PREFIX/venv"
CLONE="$PREFIX/src"
DB="$PREFIX/inferencedeck.db"
LINK="$BIN_DIR/inferencedeck"

PURGE=0
YES=0

info() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
die() { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }
usage() { sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'; }

while [ $# -gt 0 ]; do
    case "$1" in
        --purge) PURGE=1; shift ;;
        -y|--yes) YES=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown option: $1 (see --help)" ;;
    esac
done

# Refuse to rm -rf something that obviously isn't an InferenceDeck home.
case "$PREFIX" in
    ""|"/"|"$HOME"|"$HOME/") die "refusing to uninstall from '$PREFIX'" ;;
esac
if [ -e "$PREFIX" ] && [ ! -e "$VENV" ] && [ ! -e "$CLONE" ] && [ ! -e "$DB" ]; then
    die "$PREFIX doesn't look like an InferenceDeck install (no venv, src or inferencedeck.db)"
fi

# Collect what exists, so the summary only lists real paths.
targets=()
if [ -L "$LINK" ] && [ "$(readlink "$LINK")" = "$VENV/bin/inferencedeck" ]; then
    targets+=("$LINK")
fi
if [ "$PURGE" -eq 1 ]; then
    [ -e "$PREFIX" ] && targets+=("$PREFIX")
else
    [ -e "$VENV" ] && targets+=("$VENV")
    [ -e "$CLONE" ] && targets+=("$CLONE")
fi

if [ "${#targets[@]}" -eq 0 ]; then
    info "nothing to remove: InferenceDeck is not installed in $PREFIX"
    exit 0
fi

echo "This will remove:"
for t in "${targets[@]}"; do
    size="$(du -sh "$t" 2>/dev/null | cut -f1 || true)"
    printf '  %s%s\n' "$t" "${size:+  ($size)}"
done
if [ "$PURGE" -eq 1 ]; then
    [ -e "$DB" ] && echo "  (including your drafts database $DB)"
elif [ -e "$DB" ]; then
    echo "Keeping your drafts database: $DB  (use --purge to delete it)"
fi

if [ "$YES" -ne 1 ]; then
    [ -t 0 ] || die "not a terminal; re-run with --yes to confirm"
    printf 'Continue? [y/N] '
    read -r answer
    case "$answer" in
        y|Y|yes|YES) ;;
        *) info "cancelled"; exit 1 ;;
    esac
fi

for t in "${targets[@]}"; do
    rm -rf "$t"
    info "removed $t"
done
info "InferenceDeck uninstalled."
