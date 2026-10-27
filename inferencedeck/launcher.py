"""Launch drafts by shelling out to the docker CLI, so commands run exactly as written."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .drafts import Draft
from .resources import parse_docker_run, split_command


class LaunchError(RuntimeError):
    """A draft can't be launched (empty, unknown kind, docker CLI missing...)."""
    pass


@dataclass
class LaunchPlan:
    """The exact command to run for a draft, plus warnings and temp-file cleanup."""
    argv: list[str]
    cwd: str | None
    warnings: list[str]
    cleanup: Callable[[], None] = lambda: None

    @property
    def display(self) -> str:
        """Short form of the command for the launch log."""
        return " ".join(self.argv) if self.argv[:2] != ["/bin/sh", "-c"] else "sh -c <draft>"


def compose_command(docker_bin: str | None = None) -> list[str]:
    """``docker compose`` (v2 plugin) or the legacy ``docker-compose`` binary."""
    docker_bin = docker_bin or shutil.which("docker")
    if docker_bin:
        try:
            subprocess.run([docker_bin, "compose", "version"], capture_output=True, check=True, timeout=10)
            return [docker_bin, "compose"]
        except (OSError, subprocess.SubprocessError):
            pass
    legacy = shutil.which("docker-compose")
    if legacy:
        return [legacy]
    raise LaunchError("Neither 'docker compose' nor 'docker-compose' is available.")


def plan_launch(draft: Draft, *, compose_cmd: list[str] | None = None) -> LaunchPlan:
    """Turn a draft into a LaunchPlan: `sh -c <body>` for run drafts, `docker compose up -d` for compose."""
    warnings: list[str] = []
    cwd = draft.working_dir if draft.working_dir and os.path.isdir(draft.working_dir) else None
    if draft.working_dir and cwd is None:
        warnings.append(f"working dir {draft.working_dir} does not exist; relative paths may break.")

    if draft.kind == "run":
        if not draft.body.strip():
            raise LaunchError("Draft is empty.")
        if shutil.which("docker") is None:
            raise LaunchError("docker CLI not found on PATH.")
        if parse_docker_run(draft.body) is None:
            warnings.append("draft does not look like a 'docker run' command; running it anyway.")
        elif not any(t in ("-d", "--detach") or (t.startswith("-") and not t.startswith("--") and "d" in t)
                     for t in split_command(draft.body)):
            warnings.append("no -d/--detach: the command will block until the container exits.")
        return LaunchPlan(["/bin/sh", "-c", draft.body], cwd, warnings)

    if draft.kind == "compose":
        cmd = compose_cmd or compose_command()
        tmpdir = tempfile.mkdtemp(prefix="inferencedeck-")
        compose_file = Path(tmpdir) / "compose.yaml"
        compose_file.write_text(draft.body)
        project_dir = cwd or os.getcwd()
        if not draft.working_dir:
            warnings.append(f"no working dir set; relative paths resolve against {project_dir}.")
        return LaunchPlan(
            [*cmd, "-f", str(compose_file), "--project-directory", project_dir, "up", "-d"],
            project_dir,
            warnings,
            cleanup=lambda: shutil.rmtree(tmpdir, ignore_errors=True),
        )

    raise LaunchError(f"Unknown draft kind: {draft.kind}")


def run_plan(
    plan: LaunchPlan,
    on_line: Callable[[str], None],
    on_start: Callable[[subprocess.Popen], None] | None = None,
) -> int:
    """Run synchronously, streaming merged stdout/stderr lines to ``on_line``.

    ``on_start`` receives the process so the caller can terminate it.
    """
    try:
        proc = subprocess.Popen(
            plan.argv,
            cwd=plan.cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            env=os.environ.copy(),
        )
    except OSError as exc:
        on_line(f"failed to start: {exc}")
        plan.cleanup()
        return 127
    if on_start is not None:
        on_start(proc)
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            on_line(line.rstrip("\n"))
        return proc.wait()
    finally:
        plan.cleanup()
