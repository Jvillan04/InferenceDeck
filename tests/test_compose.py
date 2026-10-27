from inferencedeck.compose import compose_info_from_labels, reconstruct_docker_run
from inferencedeck.resources import GIB, parse_docker_run

from .conftest import make_attrs


def test_compose_info_reads_files(tmp_path):
    (tmp_path / "compose.yaml").write_text("services: {}\n")
    info = compose_info_from_labels(
        {
            "com.docker.compose.project": "llm",
            "com.docker.compose.service": "vllm",
            "com.docker.compose.project.working_dir": str(tmp_path),
            "com.docker.compose.project.config_files": f"compose.yaml,{tmp_path / 'override.yaml'}",
        }
    )
    assert info.project == "llm" and info.service == "vllm"
    assert info.files[0].content == "services: {}\n"
    assert info.files[1].content is None and "not found" in info.files[1].error
    assert len(info.readable) == 1


def test_no_compose_labels():
    assert compose_info_from_labels({}) is None


def test_reconstruct_masks_secrets_and_round_trips():
    attrs = make_attrs(
        "vllm-a",
        "vllm/vllm-openai:latest",
        cmd=["--model", "m", "--tensor-parallel-size", "2"],
        env=["HF_TOKEN=hf_supersecret", "PATH=/usr/bin", "VLLM_LOGGING_LEVEL=DEBUG"],
        host_config={
            "ShmSize": 16 * GIB,
            "Memory": 64 * GIB,
            "DeviceRequests": [{"DeviceIDs": ["0", "1"], "Capabilities": [["gpu"]]}],
            "PortBindings": {"8000/tcp": [{"HostIp": "", "HostPort": "8000"}]},
            "RestartPolicy": {"Name": "unless-stopped"},
        },
    )
    cmd = reconstruct_docker_run(attrs, image_env=["PATH=/usr/bin"])
    assert "hf_supersecret" not in cmd
    assert 'HF_TOKEN="${HF_TOKEN}"' in cmd
    assert "PATH=" not in cmd  # baked into the image
    assert "--model m" in cmd  # flag/value pairs stay together

    spec = parse_docker_run(cmd)
    assert spec.name == "vllm-a" and spec.image == "vllm/vllm-openai:latest"
    assert spec.limits.gpus.ids == ["0", "1"]
    assert spec.limits.shm_size == 16 * GIB and spec.limits.memory == 64 * GIB
    assert spec.argv == ["--model", "m", "--tensor-parallel-size", "2"]
    assert spec.ports == ["8000:8000"]
