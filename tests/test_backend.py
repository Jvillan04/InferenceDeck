from types import SimpleNamespace

import pytest
from docker.errors import DockerException

from inferencedeck.backend import ContainerInfo, DockerBackend, DockerUnavailable, is_llm_container


def entry(name, image, state="running", **extra):
    return {"Id": f"{name}-id".ljust(64, "0"), "Names": [f"/{name}"], "Image": image, "State": state, "Status": "Up", **extra}


class FakeAPI:
    def __init__(self, entries=None, error=None):
        self.entries = entries or []
        self.error = error

    def containers(self, all=False):
        assert all is True
        if self.error:
            raise self.error
        return self.entries


@pytest.mark.parametrize(
    "name,image,expected",
    [
        ("my-VLLM-server", "nginx", True),
        ("llm", "lmsysorg/SGLang:latest", True),
        ("llm", "vllm/vllm-openai:v0.6", True),
        ("redis", "redis:7", False),
    ],
)
def test_is_llm_container(name, image, expected):
    assert is_llm_container(name, image) is expected


def test_list_filters_and_sorts():
    api = FakeAPI(
        [
            entry("zeta-vllm", "nginx"),
            entry("redis", "redis:7"),
            entry("alpha", "lmsysorg/sglang:latest", state="exited"),
            entry("paused-one", "vllm/vllm-openai", state="paused"),
        ]
    )
    backend = DockerBackend(client=SimpleNamespace(api=api))
    infos = backend.list_llm_containers()
    assert [i.name for i in infos] == ["alpha", "paused-one", "zeta-vllm"]
    assert [i.emoji for i in infos] == ["🔴", "🟡", "🟢"]
    assert infos[0].engine == "sglang"


def test_list_raises_docker_unavailable_and_reconnects():
    backend = DockerBackend(client=SimpleNamespace(api=FakeAPI(error=DockerException("socket gone"))))
    with pytest.raises(DockerUnavailable):
        backend.list_llm_containers()
    assert backend._client is None  # next call reconnects


def test_from_list_entry_ports_and_labels():
    info = ContainerInfo.from_list_entry(
        entry(
            "vllm",
            "vllm/vllm-openai",
            Ports=[{"IP": "0.0.0.0", "PrivatePort": 8000, "PublicPort": 8000, "Type": "tcp"}, {"PrivatePort": 9000, "Type": "tcp"}],
            Labels={"com.docker.compose.project": "llm"},
        )
    )
    assert info.ports == ("0.0.0.0:8000->8000/tcp", "9000/tcp")
    assert info.compose_project == "llm"
    assert info.short_id == info.id[:12]
