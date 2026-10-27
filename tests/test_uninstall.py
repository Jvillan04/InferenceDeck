import os
import pathlib
import subprocess

import pytest

from inferencedeck import __main__ as cli

SCRIPT = pathlib.Path(__file__).resolve().parent.parent / "uninstall.sh"


@pytest.fixture
def install(tmp_path):
    """A fake install laid out like install.sh does it."""
    home = tmp_path / "home"
    prefix = home / ".local" / "share" / "inferencedeck"
    (prefix / "venv" / "bin").mkdir(parents=True)
    (prefix / "venv" / "bin" / "inferencedeck").write_text("#!/bin/sh\n")
    (prefix / "src").mkdir()
    (prefix / "inferencedeck.db").write_text("drafts")
    bin_dir = home / ".local" / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "inferencedeck").symlink_to(prefix / "venv" / "bin" / "inferencedeck")
    return home, prefix, bin_dir


def run(home, *args, stdin=subprocess.DEVNULL):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("INFERENCEDECK_", "XDG_"))}
    env["HOME"] = str(home)
    return subprocess.run(["bash", str(SCRIPT), *args], env=env, stdin=stdin, capture_output=True, text=True)


def test_removes_app_keeps_drafts(install):
    home, prefix, bin_dir = install
    result = run(home, "--yes")
    assert result.returncode == 0, result.stderr
    assert not (prefix / "venv").exists() and not (prefix / "src").exists()
    assert not (bin_dir / "inferencedeck").exists()
    assert (prefix / "inferencedeck.db").read_text() == "drafts"
    assert "Keeping your drafts database" in result.stdout


def test_purge_removes_everything(install):
    home, prefix, bin_dir = install
    result = run(home, "--purge", "--yes")
    assert result.returncode == 0, result.stderr
    assert not prefix.exists() and not (bin_dir / "inferencedeck").exists()


def test_asks_for_confirmation_when_not_a_terminal(install):
    home, prefix, _ = install
    result = run(home)
    assert result.returncode != 0 and "--yes" in result.stderr
    assert (prefix / "venv").exists()


def test_leaves_unrelated_symlink_alone(install):
    home, _, bin_dir = install
    (bin_dir / "inferencedeck").unlink()
    (bin_dir / "inferencedeck").symlink_to("/usr/bin/true")
    run(home, "--yes")
    assert (bin_dir / "inferencedeck").is_symlink()


def test_refuses_directory_that_is_not_an_install(tmp_path):
    home = tmp_path / "home"
    prefix = home / ".local" / "share" / "inferencedeck"
    prefix.mkdir(parents=True)
    (prefix / "important.txt").write_text("x")
    result = run(home, "--purge", "--yes")
    assert result.returncode != 0 and "doesn't look like" in result.stderr
    assert (prefix / "important.txt").exists()


def test_nothing_installed(tmp_path):
    result = run(tmp_path, "--yes")
    assert result.returncode == 0 and "nothing to remove" in result.stdout


def test_cli_execs_uninstall_script(monkeypatch):
    calls = []
    monkeypatch.setattr(cli.os, "execv", lambda path, argv: calls.append((path, argv)))
    cli.main(["uninstall", "--purge", "--yes"])
    assert calls == [("/bin/bash", ["bash", str(SCRIPT), "--purge", "--yes"])]
