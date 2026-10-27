from inferencedeck.lint import lint_draft, report_for_container
from inferencedeck.resources import (
    GIB,
    GpuAllotment,
    GpuInfo,
    GpuSnapshot,
    Peer,
    analyze,
    limits_from_inspect,
    parse_apps_csv,
    parse_docker_run,
    parse_engine_args,
    parse_gpu_csv,
    parse_gpus_flag,
    parse_size,
)

from .conftest import make_attrs

H100 = 81559


def snapshot(*used):
    return GpuSnapshot(gpus=[GpuInfo(i, f"GPU-{i}", "H100", H100, u) for i, u in enumerate(used)])


def severities(findings):
    return [f.severity for f in findings]


def test_parse_size():
    assert parse_size("16g") == 16 * GIB
    assert parse_size("512MB") == 512 * 1024**2
    assert parse_size("1.5GiB") == int(1.5 * GIB)
    assert parse_size(1024) == 1024
    assert parse_size("bogus") is None


def test_vllm_args_positional_model_and_equals_form():
    cfg = parse_engine_args(["vllm", "serve", "Qwen/Qwen2.5-7B", "--gpu-memory-utilization=0.8", "-tp", "2", "--max-model-len", "8192"])
    assert (cfg.engine, cfg.model, cfg.gpu_fraction, cfg.tp, cfg.max_len) == ("vllm", "Qwen/Qwen2.5-7B", 0.8, 2, 8192)
    assert not cfg.fraction_is_default


def test_vllm_default_fraction():
    cfg = parse_engine_args(["--model", "m"], hint="vllm/vllm-openai:latest")
    assert cfg.gpu_fraction == 0.9 and cfg.fraction_is_default


def test_sglang_inside_shell_wrapper():
    cfg = parse_engine_args(
        ["bash", "-c", "pip install x && python3 -m sglang.launch_server --model-path m --mem_fraction_static 0.7 --tp 4 --dp-size 2"]
    )
    assert (cfg.engine, cfg.model, cfg.gpu_fraction, cfg.tp, cfg.dp, cfg.gpus_needed) == ("sglang", "m", 0.7, 4, 2, 8)


def test_parse_gpus_flag():
    assert parse_gpus_flag("all").all
    assert parse_gpus_flag("2").count == 2
    assert parse_gpus_flag('"device=0,1"').ids == ["0", "1"]


def test_parse_docker_run():
    spec = parse_docker_run(
        """docker run -d --rm --name vllm-a \\
          --gpus '"device=1,2"' -m 64g --cpus=8 --shm-size 16g \\
          -e HF_TOKEN=x -p 8000:8000 \\
          vllm/vllm-openai:latest --model m --tensor-parallel-size 2  # trailing comment
        """
    )
    assert spec.name == "vllm-a" and spec.image == "vllm/vllm-openai:latest"
    assert spec.limits.gpus.ids == ["1", "2"]
    assert spec.limits.memory == 64 * GIB and spec.limits.cpus == 8 and spec.limits.shm_size == 16 * GIB
    assert spec.argv == ["--model", "m", "--tensor-parallel-size", "2"]
    assert parse_docker_run("echo hello") is None


def test_limits_from_inspect():
    attrs = make_attrs(
        "v",
        "vllm/vllm-openai",
        host_config={
            "Memory": 32 * GIB,
            "NanoCpus": 4_000_000_000,
            "ShmSize": 1 * GIB,
            "IpcMode": "private",
            "DeviceRequests": [{"Driver": "", "Count": 0, "DeviceIDs": ["0"], "Capabilities": [["gpu"]]}],
        },
    )
    lim = limits_from_inspect(attrs)
    assert lim.memory == 32 * GIB and lim.cpus == 4 and lim.gpus.ids == ["0"] and lim.shm_size == GIB


def test_limits_from_nvidia_runtime_env():
    attrs = make_attrs("v", "img", env=["NVIDIA_VISIBLE_DEVICES=all"], host_config={"Runtime": "nvidia"})
    assert limits_from_inspect(attrs).gpus.all


def test_nvidia_smi_parsing_and_resolution():
    snap = GpuSnapshot(
        gpus=parse_gpu_csv("0, GPU-aaa, NVIDIA H100, 81559, 1000, 5\n1, GPU-bbb, NVIDIA H100, 81559, 2000, [N/A]\n"),
        apps=parse_apps_csv("123, GPU-bbb, 1500\n456, GPU-aaa, 100\n"),
    )
    assert [g.index for g in snap.gpus] == [0, 1]
    assert snap.gpus[1].util_pct is None
    assert [g.index for g in snap.resolve(GpuAllotment(ids=["GPU-bbb"]))] == [1]
    assert [g.index for g in snap.resolve(GpuAllotment(count=1))] == [0]
    assert snap.usage_by_pids({123}) == {"GPU-bbb": 1500}


def test_analyze_tp_exceeds_gpus():
    cfg = parse_engine_args(["vllm", "serve", "m", "--tensor-parallel-size", "4"])
    spec = parse_docker_run("docker run --gpus 2 --ipc=host vllm/vllm-openai")
    msgs = [f.message for f in analyze(cfg, spec.limits, snapshot(0, 0, 0, 0)) if f.severity == "error"]
    assert any("needs 4 GPUs" in m for m in msgs)


def test_analyze_fraction_vs_free_memory():
    cfg = parse_engine_args(["python", "-m", "sglang.launch_server", "--mem-fraction-static", "0.9"])
    spec = parse_docker_run("docker run --gpus device=0 --ipc=host lmsysorg/sglang")
    errors = [f for f in analyze(cfg, spec.limits, snapshot(30000)) if f.severity == "error"]
    assert errors and "GPU 0" in errors[0].message
    assert "error" not in severities(analyze(cfg, spec.limits, snapshot(1000)))


def test_analyze_running_container_ignores_its_own_usage():
    cfg = parse_engine_args(["vllm", "serve", "m", "--gpu-memory-utilization", "0.9"])
    spec = parse_docker_run("docker run --gpus device=0 --ipc=host vllm/vllm-openai")
    findings = analyze(cfg, spec.limits, snapshot(74000), running=True, own_usage_mib={"GPU-0": 73000})
    assert "error" not in severities(findings)


def test_analyze_peers_sharing_gpu():
    cfg = parse_engine_args(["vllm", "serve", "m", "--gpu-memory-utilization", "0.6"])
    spec = parse_docker_run("docker run --gpus device=0 --ipc=host vllm/vllm-openai")
    peers = [Peer("other", GpuAllotment(ids=["0"]), 0.5), Peer("elsewhere", GpuAllotment(ids=["1"]), 0.9)]
    errors = [f.message for f in analyze(cfg, spec.limits, snapshot(0, 0), peers=peers) if f.severity == "error"]
    assert len(errors) == 1 and "'other'" in errors[0]


def test_analyze_shm_and_ram():
    cfg = parse_engine_args(["vllm", "serve", "m", "-tp", "2", "--swap-space", "16", "--cpu-offload-gb", "8"])
    spec = parse_docker_run("docker run --gpus all -m 16g vllm/vllm-openai")
    findings = analyze(cfg, spec.limits, None)
    shm = next(f for f in findings if "shm is 64 MiB" in f.message)
    ram = next(f for f in findings if "swap-space+cpu-offload" in f.message)
    assert shm.severity == "error"  # tp=2 needs shm for NCCL
    assert ram.severity == "error"


def test_no_gpus_requested():
    cfg = parse_engine_args(["vllm", "serve", "m"])
    spec = parse_docker_run("docker run --ipc=host vllm/vllm-openai")
    assert any("No GPUs requested" in f.message for f in analyze(cfg, spec.limits, None))


def test_lint_compose_draft_and_name_clash():
    body = """
services:
  proxy:
    image: nginx
  sglang:
    image: lmsysorg/sglang
    ipc: host
    command: python3 -m sglang.launch_server --model-path m --tp 2
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["0"]
              capabilities: [gpu]
"""
    reports = lint_draft("compose", body, snapshot(0))
    assert [r.scope for r in reports] == ["service sglang"]  # sidecar skipped
    assert any("needs 2 GPUs" in f.message for f in reports[0].findings)

    run = lint_draft("run", "docker run -d --name taken --gpus all --ipc=host vllm/vllm-openai --model m", existing_names=["taken"])
    assert "already exists" in run[0].findings[0].message
    assert lint_draft("compose", "services: [")[0].findings[0].severity == "error"


def test_report_for_container():
    attrs = make_attrs(
        "vllm-a",
        "vllm/vllm-openai",
        cmd=["--model", "m", "--gpu-memory-utilization", "0.5"],
        host_config={"IpcMode": "host", "DeviceRequests": [{"Count": -1, "Capabilities": [["gpu"]]}]},
    )
    report = report_for_container(attrs, snapshot(0), peers=[Peer("vllm-a", GpuAllotment(all=True), 0.9)])
    assert report.engine.gpu_fraction == 0.5
    assert "error" not in severities(report.findings)  # itself is not a peer
