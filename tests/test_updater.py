import subprocess

import pytest

from inferencedeck import updater


def git(cwd, *args):
    subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True, text=True)


@pytest.fixture
def repos(tmp_path):
    """An upstream bare repo, the 'installed' clone, and a second clone used to publish commits."""
    upstream = tmp_path / "upstream.git"
    subprocess.run(["git", "init", "--bare", "-b", "main", str(upstream)], check=True, capture_output=True)
    seed = tmp_path / "seed"
    subprocess.run(["git", "clone", str(upstream), str(seed)], check=True, capture_output=True)
    git(seed, "config", "user.email", "t@example.com")
    git(seed, "config", "user.name", "t")
    (seed / "pyproject.toml").write_text("[project]\nname='x'\n")
    git(seed, "add", ".")
    git(seed, "commit", "-m", "initial")
    git(seed, "push", "origin", "HEAD:main")
    installed = tmp_path / "installed"
    subprocess.run(["git", "clone", str(upstream), str(installed)], check=True, capture_output=True)
    return seed, installed


def test_find_repo(repos):
    _, installed = repos
    (installed / "pkg").mkdir()
    assert updater.find_repo(installed / "pkg" / "mod.py") == installed


def test_up_to_date_then_behind(repos):
    seed, installed = repos
    status = updater.check_for_updates(installed)
    assert status.error is None and status.behind == 0 and not status.available

    (seed / "new.txt").write_text("x")
    git(seed, "add", ".")
    git(seed, "commit", "-m", "add feature")
    git(seed, "push", "origin", "HEAD:main")

    status = updater.check_for_updates(installed)
    assert status.available and status.behind == 1
    assert status.commits[0].endswith("add feature")


def test_apply_refuses_dirty_checkout(repos):
    seed, installed = repos
    (seed / "new.txt").write_text("x")
    git(seed, "add", ".")
    git(seed, "commit", "-m", "more")
    git(seed, "push", "origin", "HEAD:main")
    (installed / "pyproject.toml").write_text("changed")
    lines = []
    assert updater.apply_update(installed, lines.append) == 1
    assert "local changes" in lines[-1]


def test_not_a_checkout(tmp_path, monkeypatch):
    monkeypatch.setattr(updater, "find_repo", lambda start=None: None)
    status = updater.check_for_updates()
    assert status.error and "install.sh" in status.error
