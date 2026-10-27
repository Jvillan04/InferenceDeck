"""Starter bodies for new drafts."""

VLLM_RUN = """\
docker run -d --name vllm-qwen \\
  --gpus all \\
  --ipc=host \\
  -p 8000:8000 \\
  -v ~/.cache/huggingface:/root/.cache/huggingface \\
  -e HF_TOKEN="${HF_TOKEN}" \\
  vllm/vllm-openai:latest \\
  --model Qwen/Qwen2.5-7B-Instruct \\
  --gpu-memory-utilization 0.90 \\
  --tensor-parallel-size 1 \\
  --max-model-len 32768
"""

SGLANG_RUN = """\
docker run -d --name sglang-qwen \\
  --gpus all \\
  --ipc=host \\
  -p 30000:30000 \\
  -v ~/.cache/huggingface:/root/.cache/huggingface \\
  -e HF_TOKEN="${HF_TOKEN}" \\
  lmsysorg/sglang:latest \\
  python3 -m sglang.launch_server \\
  --model-path Qwen/Qwen2.5-7B-Instruct \\
  --host 0.0.0.0 --port 30000 \\
  --mem-fraction-static 0.85 \\
  --tp-size 1
"""

VLLM_COMPOSE = """\
services:
  vllm:
    image: vllm/vllm-openai:latest
    container_name: vllm-qwen
    ipc: host
    ports:
      - "8000:8000"
    volumes:
      - ~/.cache/huggingface:/root/.cache/huggingface
    environment:
      HF_TOKEN: ${HF_TOKEN}
    command:
      - --model
      - Qwen/Qwen2.5-7B-Instruct
      - --gpu-memory-utilization
      - "0.90"
      - --tensor-parallel-size
      - "1"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
"""

SGLANG_COMPOSE = """\
services:
  sglang:
    image: lmsysorg/sglang:latest
    container_name: sglang-qwen
    ipc: host
    ports:
      - "30000:30000"
    volumes:
      - ~/.cache/huggingface:/root/.cache/huggingface
    environment:
      HF_TOKEN: ${HF_TOKEN}
    command: >
      python3 -m sglang.launch_server
      --model-path Qwen/Qwen2.5-7B-Instruct
      --host 0.0.0.0 --port 30000
      --mem-fraction-static 0.85
      --tp-size 1
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
"""

# (label, kind, body)
TEMPLATES: list[tuple[str, str, str]] = [
    ("vLLM · docker run", "run", VLLM_RUN),
    ("SGLang · docker run", "run", SGLANG_RUN),
    ("vLLM · compose", "compose", VLLM_COMPOSE),
    ("SGLang · compose", "compose", SGLANG_COMPOSE),
    ("Empty docker run", "run", "docker run -d \\\n  --name my-llm \\\n  IMAGE\n"),
    ("Empty compose", "compose", "services:\n  llm:\n    image: IMAGE\n"),
]
