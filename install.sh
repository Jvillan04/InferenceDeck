#!/usr/bin/env bash
# InferenceDeck installer.
#
#   ./install.sh                 install (or re-install) from this checkout, or clone if run elsewhere
#   ./install.sh --branch dev    track another branch
#   ./install.sh --dev           also install test dependencies
#   ./install.sh --uninstall     remove the app (keeps your drafts database)
#   ./install.sh --uninstall --purge   remove everything, including drafts
#
# Layout (override with env vars):
#   INFERENCEDECK_HOME  ${XDG_DATA_HOME:-~/.local/share}/inferencedeck   venv, source clone, drafts DB
#   INFERENCEDECK_BIN   ~/.local/bin                                       launcher symlink
#   INFERENCEDECK_REPO  git URL to clone from
set -euo pipefail

REPO_URL="${INFERENCEDECK_REPO:-https://github.com/Jvillan04/InferenceDeck.git}"
PREFIX="${INFERENCEDECK_HOME:-${XDG_DATA_HOME:-$HOME/.local/share}/inferencedeck}"
BIN_DIR="${INFERENCEDECK_BIN:-$HOME/.local/bin}"
VENV="$PREFIX/venv"
CLONE="$PREFIX/src"

BRANCH=""
DEV=0
UNINSTALL=0
PURGE=0

info() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33mwarning:\033[0m %s\n' "$*" >&2; }
die() { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

usage() { sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'; }

while [ $# -gt 0 ]; do
    case "$1" in
        --branch) [ $# -ge 2 ] || die "--branch needs a value"; BRANCH="$2"; shift 2 ;;
        --dev) DEV=1; shift ;;
        --uninstall) UNINSTALL=1; shift ;;
        --purge) PURGE=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown option: $1 (see --help)" ;;
    esac
done

# --------------------------------------------------------------------------- uninstall
if [ "$UNINSTALL" -eq 1 ]; then
    link="$BIN_DIR/inferencedeck"
    if [ -L "$link" ] && [ "$(readlink "$link")" = "$VENV/bin/inferencedeck" ]; then
        rm -f "$link"
        info "removed $link"
    fi
    if [ "$PURGE" -eq 1 ]; then
        rm -rf "$PREFIX"
        info "removed $PREFIX (including drafts database)"
    else
        rm -rf "$VENV" "$CLONE"
        info "removed $VENV and $CLONE; drafts kept in $PREFIX (use --purge to delete them)"
    fi
    exit 0
fi

# --------------------------------------------------------------------------- prerequisites
PYTHON=""
for candidate in python3.13 python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" >/dev/null 2>&1 &&
        "$candidate" -c 'import sys; sys.exit(sys.version_info < (3, 10))' 2>/dev/null; then
        PYTHON="$(command -v "$candidate")"
        break
    fi
done
[ -n "$PYTHON" ] || die "Python >= 3.10 is required."
"$PYTHON" -c 'import venv, ensurepip' 2>/dev/null ||
    die "$PYTHON cannot create virtualenvs; install the venv package (e.g. 'sudo apt install python3-venv')."
command -v git >/dev/null 2>&1 || die "git is required (updates are pulled with git)."

if ! command -v docker >/dev/null 2>&1; then
    warn "docker CLI not found: monitoring works over the socket, but launching drafts needs it."
elif ! docker info >/dev/null 2>&1; then
    warn "cannot talk to the Docker daemon as $(id -un); is it running / are you in the 'docker' group?"
fi
command -v nvidia-smi >/dev/null 2>&1 || warn "nvidia-smi not found: live GPU checks will be skipped."

# --------------------------------------------------------------------------- source
SCRIPT_DIR=""
if [ -n "${BASH_SOURCE[0]:-}" ] && [ -f "${BASH_SOURCE[0]}" ]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

if [ -n "$SCRIPT_DIR" ] && [ -d "$SCRIPT_DIR/.git" ] && grep -q 'name = "inferencedeck"' "$SCRIPT_DIR/pyproject.toml" 2>/dev/null; then
    SRC="$SCRIPT_DIR"
    info "installing from this checkout: $SRC"
    if [ -n "$BRANCH" ]; then
        git -C "$SRC" checkout "$BRANCH"
    fi
elif [ -d "$CLONE/.git" ]; then
    SRC="$CLONE"
    info "updating existing clone in $SRC"
    git -C "$SRC" fetch --quiet origin
    if [ -n "$BRANCH" ]; then
        git -C "$SRC" checkout "$BRANCH"
    fi
    git -C "$SRC" pull --ff-only --quiet
else
    SRC="$CLONE"
    info "cloning $REPO_URL into $SRC"
    mkdir -p "$PREFIX"
    git clone --quiet ${BRANCH:+--branch "$BRANCH"} "$REPO_URL" "$SRC"
fi

# --------------------------------------------------------------------------- venv + package
if [ ! -x "$VENV/bin/python" ]; then
    info "creating virtualenv in $VENV"
    mkdir -p "$PREFIX"
    "$PYTHON" -m venv "$VENV"
fi
info "installing dependencies"
"$VENV/bin/python" -m pip install --quiet --disable-pip-version-check --upgrade pip
if [ "$DEV" -eq 1 ]; then
    "$VENV/bin/python" -m pip install --quiet --disable-pip-version-check -e "${SRC}[dev]"
else
    "$VENV/bin/python" -m pip install --quiet --disable-pip-version-check -e "$SRC"
fi
"$VENV/bin/python" -m inferencedeck migrate

mkdir -p "$BIN_DIR"
ln -sfn "$VENV/bin/inferencedeck" "$BIN_DIR/inferencedeck"
info "linked $BIN_DIR/inferencedeck"

case ":$PATH:" in
    *":$BIN_DIR:"*) ;;
    *) warn "$BIN_DIR is not on your PATH; add this to your shell profile:  export PATH=\"$BIN_DIR:\$PATH\"" ;;
esac

info "installed $("$VENV/bin/inferencedeck" --version)"
info "run 'inferencedeck' to start; update later with 'inferencedeck update' or the U key in the app."
