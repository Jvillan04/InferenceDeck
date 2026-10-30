"""Generate the README screenshots by driving the real app against demo data.

    python scripts/screenshots.py            # writes docs/screenshots/*.svg
    node scripts/svg_to_png.mjs              # converts them to PNG (needs Playwright)

Everything shown is made up: no Docker daemon, GPU or real containers are used.
"""

from __future__ import annotations

import asyncio
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from inferencedeck import app as app_module  # noqa: E402
from inferencedeck.app import InferenceDeckApp  # noqa: E402
from inferencedeck.backend import ContainerInfo, is_llm_container  # noqa: E402
from inferencedeck.compose import ComposeFile, ComposeInfo  # noqa: E402
from inferencedeck.drafts import DraftStore  # noqa: E402
from inferencedeck.resources import ComputeApp, GpuInfo, GpuSnapshot  # noqa: E402
from inferencedeck.templates import SGLANG_RUN  # noqa: E402

OUT = ROOT / "docs" / "screenshots"
SIZE = (140, 40)
GIB = 1024**3

COMPOSE_FILE = """\
services:
  sglang:
    image: lmsysorg/sglang:v0.5.4
    container_name: sglang-qwen2.5-32b
    ipc: host
    ports:
      - "30000:30000"
    volumes:
      - /srv/models:/models
    environment:
      HF_TOKEN: ${HF_TOKEN}
    command: >
      python3 -m sglang.launch_server
      --model-path Qwen/Qwen2.5-32B-Instruct
      --host 0.0.0.0 --port 30000
      --tp-size 2
      --mem-fraction-static 0.85
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["2", "3"]
              capabilities: [gpu]
"""

VLLM_LOG = """\
INFO 10-30 09:12:03 [api_server.py:1885] vLLM API server version 0.11.0
INFO 10-30 09:12:03 [utils.py:328] non-default args: {'model': 'meta-llama/Llama-3.1-8B-Instruct', 'max_model_len': 32768, 'gpu_memory_utilization': 0.45}
INFO 10-30 09:12:11 [model.py:547] Resolved architecture: LlamaForCausalLM
INFO 10-30 09:12:11 [model.py:1510] Using max model len 32768
INFO 10-30 09:12:14 [core.py:644] Waiting for init message from front-end.
INFO 10-30 09:12:15 [gpu_model_runner.py:2602] Starting to load model meta-llama/Llama-3.1-8B-Instruct...
INFO 10-30 09:12:16 [cuda.py:366] Using Flash Attention backend on V1 engine.
INFO 10-30 09:12:16 [weight_utils.py:392] Using model weights format ['*.safetensors']
Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:03<00:00,  1.21it/s]
INFO 10-30 09:12:20 [default_loader.py:267] Loading weights took 3.41 seconds
INFO 10-30 09:12:21 [gpu_model_runner.py:2653] Model loading took 14.99 GiB and 4.87 seconds
INFO 10-30 09:12:31 [backends.py:548] Using cache directory: /root/.cache/vllm/torch_compile_cache/3b1f0c8a2d
INFO 10-30 09:12:38 [monitor.py:34] torch.compile takes 15.82 s in total
INFO 10-30 09:12:40 [gpu_worker.py:298] Available KV cache memory: 18.41 GiB
INFO 10-30 09:12:40 [kv_cache_utils.py:1087] GPU KV cache size: 150,816 tokens
INFO 10-30 09:12:40 [kv_cache_utils.py:1091] Maximum concurrency for 32,768 tokens per request: 4.60x
Capturing CUDA graphs (mixed prefill-decode, PIECEWISE): 100%|██████████| 67/67 [00:05<00:00, 12.84it/s]
INFO 10-30 09:12:47 [gpu_model_runner.py:3480] Graph capturing finished in 6 secs, took 0.52 GiB
INFO 10-30 09:12:47 [core.py:210] init engine (profile, create kv cache, warmup model) took 26.13 seconds
INFO 10-30 09:12:48 [api_server.py:1680] Starting vLLM API server 0 on http://0.0.0.0:8000
INFO 10-30 09:12:48 [launcher.py:44] Route: /v1/chat/completions, Methods: POST
INFO:     Started server process [1]
INFO:     Application startup complete.
INFO 10-30 09:14:02 [loggers.py:127] Engine 000: Avg prompt throughput: 812.4 tokens/s, Avg generation throughput: 1874.9 tokens/s, Running: 9 reqs, Waiting: 0 reqs, GPU KV cache usage: 21.7%, Prefix cache hit rate: 63.2%
INFO:     172.17.0.1:51034 - "POST /v1/chat/completions HTTP/1.1" 200 OK
INFO:     172.17.0.1:51040 - "POST /v1/chat/completions HTTP/1.1" 200 OK
INFO 10-30 09:14:12 [loggers.py:127] Engine 000: Avg prompt throughput: 640.1 tokens/s, Avg generation throughput: 1921.3 tokens/s, Running: 11 reqs, Waiting: 0 reqs, GPU KV cache usage: 26.4%, Prefix cache hit rate: 65.8%
"""


def gpu_request(*ids: str) -> list[dict]:
    return [{"Driver": "nvidia", "Count": 0, "DeviceIDs": list(ids), "Capabilities": [["gpu"]]}]


def container(cid, name, image, state, cmd, *, gpus, entrypoint=None, labels=None, shm=None, ipc="host",
              memory=0, ports=None):
    """A docker inspect payload with the fields InferenceDeck reads."""
    return {
        "Id": cid.ljust(64, "0"),
        "Name": f"/{name}",
        "State": {"Status": state, "StartedAt": "2026-10-30T09:12:01.000Z", "ExitCode": 0 if state != "exited" else 137},
        "RestartCount": 0,
        "Config": {"Image": image, "Entrypoint": entrypoint, "Cmd": cmd, "Labels": labels or {},
                   "Env": ["HF_TOKEN=hf_demo", "PATH=/usr/local/bin:/usr/bin"]},
        "HostConfig": {"DeviceRequests": gpu_request(*gpus), "IpcMode": ipc, "ShmSize": shm or 64 * 1024**2,
                       "Memory": memory, "PortBindings": {f"{p}/tcp": [{"HostIp": "", "HostPort": str(p)}] for p in ports or []}},
        "Mounts": [{"Type": "bind", "Source": "/srv/models", "Destination": "/root/.cache/huggingface", "RW": True}],
        "NetworkSettings": {"Ports": {f"{p}/tcp": [{"HostIp": "0.0.0.0", "HostPort": str(p)}] for p in ports or []}},
    }


CONTAINERS = [
    container("a1b2c3d4e5f6", "vllm-llama-3.1-8b", "vllm/vllm-openai:v0.11.0", "running",
              ["--model", "meta-llama/Llama-3.1-8B-Instruct", "--gpu-memory-utilization", "0.45", "--max-model-len", "32768"],
              gpus=["0"], ports=[8000], memory=64 * GIB),
    container("b2c3d4e5f6a1", "sglang-qwen2.5-32b", "lmsysorg/sglang:v0.5.4", "running",
              ["python3", "-m", "sglang.launch_server", "--model-path", "Qwen/Qwen2.5-32B-Instruct", "--host", "0.0.0.0",
               "--port", "30000", "--tp-size", "2", "--mem-fraction-static", "0.85"],
              gpus=["2", "3"], ports=[30000],
              labels={"com.docker.compose.project": "qwen", "com.docker.compose.service": "sglang",
                      "com.docker.compose.project.working_dir": "/srv/llm/qwen",
                      "com.docker.compose.project.config_files": "/srv/llm/qwen/compose.yaml"}),
    container("c3d4e5f6a1b2", "vllm-gemma-3-27b", "vllm/vllm-openai:v0.11.0", "exited",
              ["--model", "google/gemma-3-27b-it", "--tensor-parallel-size", "2", "--gpu-memory-utilization", "0.92",
               "--max-model-len", "65536"],
              gpus=["0", "1"], ipc="private", shm=512 * 1024**2, ports=[8001]),
    container("d4e5f6a1b2c3", "vllm-mistral-small-24b", "vllm/vllm-openai:latest", "paused",
              ["--model", "mistralai/Mistral-Small-3.2-24B-Instruct-2506", "--gpu-memory-utilization", "0.5"],
              gpus=["1"], ports=[8002]),
    container("e5f6a1b2c3d4", "sglang-deepseek-r1-distill", "lmsysorg/sglang:latest", "exited",
              ["python3", "-m", "sglang.launch_server", "--model-path", "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
               "--mem-fraction-static", "0.8"],
              gpus=["1"], ports=[30001]),
]

PIDS = {"a1b2c3d4e5f6": {4101}, "b2c3d4e5f6a1": {4202, 4203}}

SNAPSHOT = GpuSnapshot(
    gpus=[
        GpuInfo(0, "GPU-0", "NVIDIA H100 80GB HBM3", 81559, 37420, 41),
        GpuInfo(1, "GPU-1", "NVIDIA H100 80GB HBM3", 81559, 412, 0),
        GpuInfo(2, "GPU-2", "NVIDIA H100 80GB HBM3", 81559, 70114, 87),
        GpuInfo(3, "GPU-3", "NVIDIA H100 80GB HBM3", 81559, 69872, 86),
    ],
    apps=[ComputeApp(4101, "GPU-0", 36900), ComputeApp(4202, "GPU-2", 69600), ComputeApp(4203, "GPU-3", 69400)],
)


class DemoStream:
    def __init__(self, text: str) -> None:
        self._lines = [line.encode() + b"\n" for line in text.splitlines()]

        self._closed = threading.Event()

    def __iter__(self):
        yield from self._lines
        self._closed.wait()  # stay open like a running container's follow stream

    def close(self) -> None:
        self._closed.set()


class DemoBackend:
    """Read-only stand-in for DockerBackend serving the demo containers."""

    def __init__(self) -> None:
        self.containers = {c["Id"]: c for c in CONTAINERS}

    def list_llm_containers(self) -> list[ContainerInfo]:
        infos = [ContainerInfo.from_inspect(a) for a in self.containers.values()]
        return sorted((i for i in infos if is_llm_container(i.name, i.image)), key=lambda i: i.name.lower())

    def inspect(self, cid: str) -> dict:
        return self.containers[cid]

    def image_env(self, image: str) -> list[str]:
        return ["PATH=/usr/local/bin:/usr/bin"]

    def top_pids(self, cid: str) -> set[int]:
        return PIDS.get(cid[:12], set())

    def open_log_stream(self, cid: str, tail: int = 200) -> DemoStream:
        return DemoStream(VLLM_LOG)


def demo_compose_info(labels):
    if not labels or "com.docker.compose.project" not in labels:
        return None
    return ComposeInfo(
        project=labels["com.docker.compose.project"],
        service=labels.get("com.docker.compose.service"),
        working_dir=labels.get("com.docker.compose.project.working_dir"),
        files=[ComposeFile(path=labels["com.docker.compose.project.config_files"], content=COMPOSE_FILE)],
    )


async def select(app, pilot, name: str, tab: str) -> None:
    names = [item.info.name for item in app.list_view.query(app_module.ContainerItem)]
    app.list_view.index = names.index(name)
    await pilot.pause(0.8)
    app.query_one(app_module.TabbedContent).active = tab
    app.list_view.focus()
    await pilot.pause(0.5)


async def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    app_module.compose_info_from_labels = demo_compose_info
    store = DraftStore(":memory:")
    draft = store.create("qwen3-30b-a3b-tp2", "run", SGLANG_RUN.replace("Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen3-30B-A3B")
                         .replace("--tp-size 1", "--tp-size 2").replace("sglang-qwen", "sglang-qwen3-moe")
                         .replace("--gpus all", "--gpus '\"device=0,1\"'"),
                         working_dir="/srv/llm/qwen3")
    store.create("llama-8b-long-context", "run", "docker run -d --name vllm-llama-long \\\n  --gpus all --ipc=host \\\n"
                 "  vllm/vllm-openai:v0.11.0 \\\n  --model meta-llama/Llama-3.1-8B-Instruct --max-model-len 131072\n")
    store.create("qwen-32b-compose", "compose", COMPOSE_FILE, working_dir="/srv/llm/qwen")

    app = InferenceDeckApp(backend=DemoBackend(), store=store, gpu_query=lambda: SNAPSHOT, check_updates=False)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause(1.5)
        await select(app, pilot, "vllm-llama-3.1-8b", "tab-details")
        app.save_screenshot("overview.svg", str(OUT))

        await select(app, pilot, "vllm-gemma-3-27b", "tab-resources")
        app.save_screenshot("resources.svg", str(OUT))

        await select(app, pilot, "vllm-llama-3.1-8b", "tab-details")
        await pilot.press("enter")
        await pilot.pause(1.0)
        app.save_screenshot("logs.svg", str(OUT))

        await select(app, pilot, "sglang-qwen2.5-32b", "tab-compose")
        app.save_screenshot("compose.svg", str(OUT))

        await pilot.press("d")
        await pilot.pause(0.5)
        await app.screen.reload_list(select_id=draft.id)
        await pilot.pause(0.8)
        app.save_screenshot("drafts.svg", str(OUT))
    print(f"wrote {len(list(OUT.glob('*.svg')))} screenshots to {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
