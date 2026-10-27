import threading

from inferencedeck.app import ContainerItem, InferenceDeckApp
from inferencedeck.resources import GpuSnapshot
from inferencedeck.screens import ConfirmScreen, DraftsScreen

from .conftest import FakeBackend, make_attrs


def make_app(backend, store):
    return InferenceDeckApp(
        backend=backend,
        store=store,
        gpu_query=lambda: GpuSnapshot(error="nvidia-smi not found"),
        poll_interval=0.2,
        check_updates=False,
    )


def labels(app):
    return {item.info.name: str(item.label.render()) for item in app.list_view.query(ContainerItem)}


async def wait_for(pilot, predicate, timeout=5.0):
    for _ in range(int(timeout / 0.05)):
        if predicate():
            return True
        await pilot.pause(0.05)
    return predicate()


async def test_lists_only_llm_containers(store):
    backend = FakeBackend(
        [make_attrs("vllm-a", "vllm/vllm-openai"), make_attrs("redis", "redis:7"), make_attrs("b", "lmsysorg/sglang", state="exited")]
    )
    app = make_app(backend, store)
    async with app.run_test() as pilot:
        assert await wait_for(pilot, lambda: len(labels(app)) == 2)
        assert labels(app)["vllm-a"].startswith("🟢") and labels(app)["b"].startswith("🔴")


async def test_stop_confirm_pending_then_updated(store):
    attrs = make_attrs("vllm-a", "vllm/vllm-openai")
    backend = FakeBackend([attrs])
    backend.gate = threading.Event()
    app = make_app(backend, store)
    async with app.run_test() as pilot:
        await wait_for(pilot, lambda: labels(app))
        await pilot.press("t")
        assert isinstance(app.screen, ConfirmScreen)
        await pilot.press("y")
        assert await wait_for(pilot, lambda: "stopping" in labels(app)["vllm-a"])  # immediate feedback
        await pilot.pause(0.5)  # polls must not overwrite the pending label
        assert "stopping" in labels(app)["vllm-a"]
        backend.gate.set()
        assert await wait_for(pilot, lambda: labels(app)["vllm-a"].startswith("🔴"))
        assert backend.calls == [("stop", attrs["Id"])]

        await pilot.press("s")  # start needs no confirmation
        assert await wait_for(pilot, lambda: labels(app)["vllm-a"].startswith("🟢"))
        await pilot.press("r", "y")
        assert await wait_for(pilot, lambda: backend.calls[-1] == ("restart", attrs["Id"]))


async def test_cancel_confirm_does_nothing(store):
    backend = FakeBackend([make_attrs("vllm-a", "vllm/vllm-openai")])
    app = make_app(backend, store)
    async with app.run_test() as pilot:
        await wait_for(pilot, lambda: labels(app))
        await pilot.press("t", "n")
        await pilot.pause(0.3)
        assert backend.calls == []


async def test_start_on_running_is_noop(store):
    backend = FakeBackend([make_attrs("vllm-a", "vllm/vllm-openai")])
    app = make_app(backend, store)
    async with app.run_test() as pilot:
        await wait_for(pilot, lambda: labels(app))
        await pilot.press("s")
        await pilot.pause(0.3)
        assert backend.calls == []


async def test_cursor_follows_container_id_when_list_changes(store):
    backend = FakeBackend([make_attrs("m-vllm", "vllm/vllm-openai"), make_attrs("z-vllm", "vllm/vllm-openai")])
    app = make_app(backend, store)
    async with app.run_test() as pilot:
        await wait_for(pilot, lambda: len(labels(app)) == 2)
        app.list_view.index = 1  # z-vllm
        await pilot.pause(0.1)
        backend.add(make_attrs("a-vllm", "vllm/vllm-openai"))  # sorts above the selection
        assert await wait_for(pilot, lambda: len(labels(app)) == 3 and not app._scan_running)
        assert app.highlighted_info().name == "z-vllm"


async def test_logs_attach_and_switch(store):
    backend = FakeBackend([make_attrs("a-vllm", "vllm/vllm-openai"), make_attrs("b-vllm", "vllm/vllm-openai")])
    app = make_app(backend, store)
    async with app.run_test() as pilot:
        await wait_for(pilot, lambda: len(labels(app)) == 2)
        await pilot.press("enter")
        log = app.query_one("#log-output")
        assert await wait_for(pilot, lambda: "a-vllm line 2" in log.lines)
        await pilot.press("down", "enter")
        assert await wait_for(pilot, lambda: "b-vllm line 2" in log.lines)
        assert backend.streams[0].closed.is_set()  # previous stream was closed
        assert "a-vllm line 1" not in log.lines


async def test_save_as_draft_and_open_drafts(store):
    backend = FakeBackend([make_attrs("vllm-a", "vllm/vllm-openai", cmd=["--model", "m"], env=["HF_TOKEN=secret"])])
    app = make_app(backend, store)
    async with app.run_test() as pilot:
        await wait_for(pilot, lambda: labels(app))
        await pilot.press("e")
        assert await wait_for(pilot, lambda: type(app.screen).__name__ == "DraftFormScreen")
        await pilot.press("enter")
        assert await wait_for(pilot, lambda: isinstance(app.screen, DraftsScreen))
        (draft,) = store.list_drafts()
        assert draft.kind == "run" and draft.source_container == "vllm-a"
        assert "secret" not in draft.body and "--model m" in draft.body
