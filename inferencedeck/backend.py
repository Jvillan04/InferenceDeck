"""Docker access for InferenceDeck.

Everything here is synchronous and free of Textual imports: the UI calls it
through ``asyncio.to_thread`` or thread workers so the event loop never blocks.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Iterator

import docker
from docker.errors import DockerException

MATCH_KEYWORDS = ("vllm", "sglang")

STATE_EMOJI = {
    "running": "🟢",
    "paused": "🟡",
    "restarting": "🔄",
    "created": "⚪",
}


class DockerUnavailable(RuntimeError):
    """Raised when the Docker daemon cannot be reached."""


def is_llm_container(name: str, image: str) -> bool:
    """A container qualifies if its name or image ref mentions vllm/sglang."""
    haystack = f"{name} {image}".lower()
    return any(k in haystack for k in MATCH_KEYWORDS)


def detect_engine(*texts: str) -> str | None:
    haystack = " ".join(texts).lower()
    if "sglang" in haystack:
        return "sglang"
    if "vllm" in haystack:
        return "vllm"
    return None


@dataclass(frozen=True)
class ContainerInfo:
    id: str
    name: str
    image: str
    state: str
    status_text: str = ""
    command: str = ""
    created: int = 0
    ports: tuple[str, ...] = ()
    labels: dict[str, str] = field(default_factory=dict, compare=False, hash=False)

    @property
    def short_id(self) -> str:
        return self.id[:12]

    @property
    def emoji(self) -> str:
        return STATE_EMOJI.get(self.state, "🔴")

    @property
    def engine(self) -> str | None:
        return detect_engine(self.image, self.name, self.command)

    @property
    def compose_project(self) -> str | None:
        return self.labels.get("com.docker.compose.project")

    @classmethod
    def from_list_entry(cls, entry: dict[str, Any]) -> "ContainerInfo":
        """Build from a low-level ``/containers/json`` entry."""
        names = entry.get("Names") or []
        name = names[0].lstrip("/") if names else entry.get("Id", "")[:12]
        ports = []
        for p in entry.get("Ports") or []:
            if p.get("PublicPort"):
                host = p.get("IP") or "0.0.0.0"
                ports.append(f"{host}:{p['PublicPort']}->{p['PrivatePort']}/{p.get('Type', 'tcp')}")
            else:
                ports.append(f"{p.get('PrivatePort')}/{p.get('Type', 'tcp')}")
        return cls(
            id=entry["Id"],
            name=name,
            image=entry.get("Image", ""),
            state=(entry.get("State") or "unknown").lower(),
            status_text=entry.get("Status", ""),
            command=entry.get("Command", ""),
            created=int(entry.get("Created") or 0),
            ports=tuple(sorted(set(ports))),
            labels=dict(entry.get("Labels") or {}),
        )

    @classmethod
    def from_inspect(cls, attrs: dict[str, Any]) -> "ContainerInfo":
        """Build from a full ``inspect`` payload (``Container.attrs``)."""
        config = attrs.get("Config") or {}
        state = attrs.get("State") or {}
        cmd = (config.get("Entrypoint") or []) + (config.get("Cmd") or [])
        ports = []
        for private, bindings in ((attrs.get("NetworkSettings") or {}).get("Ports") or {}).items():
            for b in bindings or []:
                ports.append(f"{b.get('HostIp') or '0.0.0.0'}:{b.get('HostPort')}->{private}")
        return cls(
            id=attrs["Id"],
            name=(attrs.get("Name") or "").lstrip("/"),
            image=config.get("Image", ""),
            state=(state.get("Status") or "unknown").lower(),
            status_text=state.get("Status", ""),
            command=" ".join(cmd),
            created=0,
            ports=tuple(sorted(set(ports))),
            labels=dict(config.get("Labels") or {}),
        )


class DockerBackend:
    """Thin, thread-safe wrapper around the Docker SDK."""

    def __init__(self, client: Any | None = None) -> None:
        self._client = client
        self._lock = threading.Lock()

    @property
    def client(self) -> Any:
        with self._lock:
            if self._client is None:
                try:
                    self._client = docker.from_env()
                except DockerException as exc:
                    raise DockerUnavailable(str(exc)) from exc
            return self._client

    def _reset_on_failure(self, exc: Exception) -> None:
        # Drop the client so the next call reconnects (daemon restarted, socket changed...).
        if isinstance(exc, DockerException) and not isinstance(exc, docker.errors.APIError):
            with self._lock:
                self._client = None

    # -- discovery -----------------------------------------------------------------

    def list_llm_containers(self) -> list[ContainerInfo]:
        try:
            entries = self.client.api.containers(all=True)
        except DockerException as exc:
            self._reset_on_failure(exc)
            raise DockerUnavailable(str(exc)) from exc
        infos = [ContainerInfo.from_list_entry(e) for e in entries]
        infos = [i for i in infos if is_llm_container(i.name, i.image)]
        return sorted(infos, key=lambda i: i.name.lower())

    def inspect(self, container_id: str) -> dict[str, Any]:
        return self.client.api.inspect_container(container_id)

    def image_env(self, image: str) -> list[str]:
        """Env baked into an image, used to tell user-set env apart. Best effort."""
        try:
            return (self.client.api.inspect_image(image).get("Config") or {}).get("Env") or []
        except DockerException:
            return []

    def top_pids(self, container_id: str) -> set[int]:
        """Host PIDs of processes inside a running container."""
        try:
            top = self.client.api.top(container_id)
        except DockerException:
            return set()
        titles = top.get("Titles") or []
        if "PID" not in titles:
            return set()
        idx = titles.index("PID")
        pids = set()
        for row in top.get("Processes") or []:
            try:
                pids.add(int(row[idx]))
            except (ValueError, IndexError):
                pass
        return pids

    # -- lifecycle -----------------------------------------------------------------

    def _fresh(self, container: Any) -> ContainerInfo:
        container.reload()
        return ContainerInfo.from_inspect(container.attrs)

    def start(self, container_id: str) -> ContainerInfo:
        container = self.client.containers.get(container_id)
        if container.status == "paused":
            container.unpause()
        else:
            container.start()
        return self._fresh(container)

    def stop(self, container_id: str, timeout: int = 30) -> ContainerInfo:
        container = self.client.containers.get(container_id)
        container.stop(timeout=timeout)
        return self._fresh(container)

    def restart(self, container_id: str, timeout: int = 30) -> ContainerInfo:
        container = self.client.containers.get(container_id)
        container.restart(timeout=timeout)
        return self._fresh(container)

    # -- logs ----------------------------------------------------------------------

    def open_log_stream(self, container_id: str, tail: int = 200) -> Iterator[bytes]:
        """Returns docker-py's CancellableStream; call ``.close()`` to unblock readers."""
        container = self.client.containers.get(container_id)
        return container.logs(stream=True, follow=True, tail=tail)
