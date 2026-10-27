from __future__ import annotations

import threading
from typing import Any

import pytest

from inferencedeck.backend import ContainerInfo, is_llm_container
from inferencedeck.drafts import DraftStore


def make_attrs(
    name: str,
    image: str,
    *,
    cid: str | None = None,
    state: str = "running",
    entrypoint: list[str] | None = None,
    cmd: list[str] | None = None,
    env: list[str] | None = None,
    labels: dict[str, str] | None = None,
    host_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "Id": cid or (name.encode().hex() * 8)[:64],
        "Name": f"/{name}",
        "State": {"Status": state, "StartedAt": "2026-10-27T10:00:00.000Z", "ExitCode": 0},
        "RestartCount": 0,
        "Config": {
            "Image": image,
            "Entrypoint": entrypoint,
            "Cmd": cmd or [],
            "Env": env or [],
            "Labels": labels or {},
        },
        "HostConfig": host_config or {},
        "Mounts": [],
        "NetworkSettings": {"Ports": {}},
    }


class FakeStream:
    def __init__(self, lines: list[bytes]) -> None:
        self._lines = list(lines)
        self.closed = threading.Event()

    def __iter__(self):
        yield from self._lines
        self.closed.wait(5)  # behave like a follow stream until closed

    def close(self) -> None:
        self.closed.set()


class FakeBackend:
    """In-memory stand-in for DockerBackend."""

    def __init__(self, containers: list[dict[str, Any]]) -> None:
        self.containers = {c["Id"]: c for c in containers}
        self.calls: list[tuple[str, str]] = []
        self.gate: threading.Event | None = None  # when set, lifecycle calls block until released
        self.streams: list[FakeStream] = []

    def add(self, attrs: dict[str, Any]) -> None:
        self.containers[attrs["Id"]] = attrs

    def _info(self, attrs: dict[str, Any]) -> ContainerInfo:
        return ContainerInfo.from_inspect(attrs)

    def list_llm_containers(self) -> list[ContainerInfo]:
        infos = [self._info(a) for a in self.containers.values()]
        return sorted((i for i in infos if is_llm_container(i.name, i.image)), key=lambda i: i.name.lower())

    def inspect(self, cid: str) -> dict[str, Any]:
        return self.containers[cid]

    def image_env(self, image: str) -> list[str]:
        return []

    def top_pids(self, cid: str) -> set[int]:
        return set()

    def _set_state(self, verb: str, cid: str, state: str) -> ContainerInfo:
        self.calls.append((verb, cid))
        if self.gate is not None:
            self.gate.wait(5)
        self.containers[cid]["State"]["Status"] = state
        return self._info(self.containers[cid])

    def start(self, cid: str) -> ContainerInfo:
        return self._set_state("start", cid, "running")

    def stop(self, cid: str, timeout: int = 30) -> ContainerInfo:
        return self._set_state("stop", cid, "exited")

    def restart(self, cid: str, timeout: int = 30) -> ContainerInfo:
        return self._set_state("restart", cid, "running")

    def open_log_stream(self, cid: str, tail: int = 200) -> FakeStream:
        name = self.containers[cid]["Name"].lstrip("/")
        stream = FakeStream([f"{name} line 1\n".encode(), f"{name} line 2\n".encode()])
        self.streams.append(stream)
        return stream


@pytest.fixture
def store(tmp_path) -> DraftStore:
    s = DraftStore(tmp_path / "deck.db")
    yield s
    s.close()
