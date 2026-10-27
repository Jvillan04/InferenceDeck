import time

import pytest

from inferencedeck import app as app_module
from inferencedeck.app import InferenceDeckApp
from inferencedeck.resources import GpuSnapshot
from inferencedeck.updater import UpdateStatus

from .conftest import FakeBackend


@pytest.fixture
def fake_updater(monkeypatch):
    """Stub check_for_updates; records the `fetch` argument of each call."""
    calls = []
    state = {"local": UpdateStatus(repo=None), "remote": UpdateStatus(repo=None)}

    def check(repo=None, *, fetch=True):
        calls.append(fetch)
        return state["remote" if fetch else "local"]

    monkeypatch.setattr(app_module.updater, "check_for_updates", check)
    return calls, state


def behind(n):
    return UpdateStatus(repo=None, branch="main", upstream="origin/main", behind=n, commits=["abc fix"] * n)


async def start(store, pilot_fn):
    app = InferenceDeckApp(
        backend=FakeBackend([]), store=store, gpu_query=lambda: GpuSnapshot(error="x"), check_updates=True
    )
    notes = []
    app.notify = lambda message, **kw: notes.append(message)  # type: ignore[method-assign]
    async with app.run_test() as pilot:
        await pilot.pause(0.5)
        await pilot_fn(app, pilot)
    return app, notes


async def test_first_run_fetches_and_shows_lasting_marker(store, fake_updater):
    calls, state = fake_updater
    state["remote"] = behind(2)

    async def check(app, pilot):
        assert "update available" in app.sub_title

    _, notes = await start(store, check)
    assert calls == [False, True]
    assert len(notes) == 1 and "2 new commit(s)" in notes[0]
    assert float(store.get_meta("last_update_check")) > time.time() - 60


async def test_known_update_is_shown_again_without_fetching(store, fake_updater):
    calls, state = fake_updater
    store.set_meta("last_update_check", str(time.time()))  # fetched recently
    state["local"] = behind(1)  # ...and that fetch found an update

    async def check(app, pilot):
        assert "update available" in app.sub_title

    _, notes = await start(store, check)
    assert calls == [False]  # no network fetch
    assert len(notes) == 1


async def test_failed_fetch_is_retried_next_time(store, fake_updater):
    calls, state = fake_updater
    state["remote"] = UpdateStatus(repo=None, error="network down")

    async def check(app, pilot):
        assert app.sub_title == app.SUB_TITLE  # errors stay silent

    _, notes = await start(store, check)
    assert calls == [False, True] and notes == []
    assert store.get_meta("last_update_check") is None


async def test_repeat_checks_do_not_renotify(store, fake_updater):
    calls, state = fake_updater
    state["local"] = state["remote"] = behind(3)

    async def check(app, pilot):
        app.quiet_update_check()
        await pilot.pause(0.3)

    _, notes = await start(store, check)
    assert len(notes) == 1
