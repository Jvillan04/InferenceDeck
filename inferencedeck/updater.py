"""Self-update support: InferenceDeck is installed as an editable install of a git checkout
(see ``install.sh``), so updating is ``git pull --ff-only`` + ``pip install -e`` + DB migrations.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import __version__


def find_repo(start: Path | None = None) -> Path | None:
    path = (start or Path(__file__)).resolve()
    for candidate in [path, *path.parents]:
        if (candidate / ".git").exists() and (candidate / "pyproject.toml").exists():
            return candidate
    return None


def _git(repo: Path, *args: str, timeout: int = 60) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True, timeout=timeout
    ).stdout.strip()


def version_string(repo: Path | None = None) -> str:
    repo = repo or find_repo()
    if repo is None:
        return __version__
    try:
        return f"{__version__} ({_git(repo, 'rev-parse', '--short', 'HEAD')})"
    except (OSError, subprocess.SubprocessError):
        return __version__


@dataclass
class UpdateStatus:
    repo: Path | None
    branch: str | None = None
    upstream: str | None = None
    behind: int = 0
    ahead: int = 0
    commits: list[str] = field(default_factory=list)
    dirty: bool = False
    error: str | None = None

    @property
    def available(self) -> bool:
        return self.error is None and self.behind > 0

    def summary(self) -> str:
        if self.error:
            return f"Update check failed: {self.error}"
        if not self.behind:
            return f"Up to date ({self.branch} @ {self.upstream})."
        return f"{self.behind} new commit(s) on {self.upstream}."


def check_for_updates(repo: Path | None = None, *, fetch: bool = True) -> UpdateStatus:
    repo = repo or find_repo()
    status = UpdateStatus(repo=repo)
    if repo is None:
        status.error = "not running from a git checkout (install with install.sh to enable updates)"
        return status
    try:
        if fetch:
            _git(repo, "fetch", "--quiet", timeout=120)
        status.branch = _git(repo, "rev-parse", "--abbrev-ref", "HEAD")
        status.upstream = _git(repo, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}")
        ahead, behind = _git(repo, "rev-list", "--left-right", "--count", "HEAD...@{u}").split()
        status.ahead, status.behind = int(ahead), int(behind)
        if status.behind:
            status.commits = _git(repo, "log", "--oneline", "--no-decorate", "HEAD..@{u}").splitlines()
        status.dirty = bool(_git(repo, "status", "--porcelain", "--untracked-files=no"))
    except subprocess.CalledProcessError as exc:
        status.error = (exc.stderr or str(exc)).strip().splitlines()[-1] if (exc.stderr or "").strip() else str(exc)
    except (OSError, subprocess.SubprocessError) as exc:
        status.error = str(exc)
    return status


def _stream(cmd: list[str], on_line: Callable[[str], None], cwd: Path | None = None) -> int:
    on_line(f"$ {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, stdin=subprocess.DEVNULL
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        on_line(line.rstrip("\n"))
    return proc.wait()


def apply_update(
    repo: Path | None = None,
    on_line: Callable[[str], None] = print,
    *,
    python: str | None = None,
) -> int:
    """Pull, reinstall into the running interpreter's environment, migrate. Returns an exit code."""
    status = check_for_updates(repo)
    if status.error:
        on_line(status.summary())
        return 1
    assert status.repo is not None
    if status.dirty:
        on_line(f"Refusing to update: {status.repo} has local changes (commit or stash them first).")
        return 1
    if not status.behind:
        on_line(status.summary())
        return 0
    on_line(status.summary())
    for c in status.commits:
        on_line(f"  {c}")
    python = python or sys.executable
    steps = [
        ["git", "-C", str(status.repo), "pull", "--ff-only"],
        [python, "-m", "pip", "install", "--quiet", "--disable-pip-version-check", "-e", str(status.repo)],
        [python, "-m", "inferencedeck", "migrate"],
    ]
    for cmd in steps:
        code = _stream(cmd, on_line)
        if code != 0:
            on_line(f"Step failed with exit code {code}.")
            return code
    on_line(f"Updated to {version_string(status.repo)}. Restart InferenceDeck to use the new version.")
    return 0
