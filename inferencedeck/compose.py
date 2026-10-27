"""Compose file discovery and ``docker run`` reconstruction from ``docker inspect``."""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

LABEL_PROJECT = "com.docker.compose.project"
LABEL_SERVICE = "com.docker.compose.service"
LABEL_FILES = "com.docker.compose.project.config_files"
LABEL_WORKDIR = "com.docker.compose.project.working_dir"

SECRET_KEY_RE = re.compile(r"TOKEN|KEY|SECRET|PASSWORD|PASSWD|CREDENTIAL", re.IGNORECASE)


@dataclass
class ComposeFile:
    path: str
    content: str | None = None
    error: str | None = None


@dataclass
class ComposeInfo:
    project: str
    service: str | None
    working_dir: str | None
    files: list[ComposeFile] = field(default_factory=list)

    @property
    def readable(self) -> list[ComposeFile]:
        return [f for f in self.files if f.content is not None]


def compose_info_from_labels(labels: dict[str, str] | None) -> ComposeInfo | None:
    labels = labels or {}
    project = labels.get(LABEL_PROJECT)
    if not project:
        return None
    workdir = labels.get(LABEL_WORKDIR)
    info = ComposeInfo(project=project, service=labels.get(LABEL_SERVICE), working_dir=workdir)
    for raw in filter(None, (p.strip() for p in labels.get(LABEL_FILES, "").split(","))):
        path = Path(raw)
        if not path.is_absolute() and workdir:
            path = Path(workdir) / path
        cf = ComposeFile(path=str(path))
        try:
            cf.content = path.read_text()
        except FileNotFoundError:
            cf.error = "not found on this host (created elsewhere, or moved/deleted)"
        except PermissionError:
            cf.error = "permission denied"
        except OSError as exc:
            cf.error = str(exc)
        info.files.append(cf)
    return info


def load_compose_services(text: str) -> dict[str, dict[str, Any]]:
    """Parse compose YAML, returning ``{service_name: service_dict}``. Raises ``ValueError``."""
    try:
        doc = yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML: {exc}") from exc
    if not isinstance(doc, dict):
        raise ValueError("compose file must be a mapping")
    services = doc.get("services") or {}
    if not isinstance(services, dict):
        raise ValueError("'services' must be a mapping")
    return {str(k): (v or {}) for k, v in services.items() if isinstance(v or {}, dict)}


# --------------------------------------------------------------------------------
# docker run reconstruction
# --------------------------------------------------------------------------------

DEFAULT_SHM = 64 * 1024 * 1024


def _size_flag(n: int) -> str:
    for unit, mult in (("g", 1024**3), ("m", 1024**2), ("k", 1024)):
        if n % mult == 0:
            return f"{n // mult}{unit}"
    return str(n)


def _q(value: str) -> str:
    return shlex.quote(value)


def reconstruct_docker_run(attrs: dict[str, Any], image_env: list[str] | None = None) -> str:
    """Best-effort ``docker run`` command equivalent to an existing container."""
    cfg = attrs.get("Config") or {}
    hc = attrs.get("HostConfig") or {}
    name = (attrs.get("Name") or "").lstrip("/")
    lines: list[str] = ["docker run -d"]

    def add(*parts: str) -> None:
        lines.append(" ".join(parts))

    if name:
        add("--name", _q(name))
    for req in hc.get("DeviceRequests") or []:
        if req.get("DeviceIDs"):
            add("--gpus", _q(f'"device={",".join(req["DeviceIDs"])}"'))
        elif req.get("Count") == -1:
            add("--gpus", "all")
        elif req.get("Count"):
            add("--gpus", str(req["Count"]))
    if hc.get("Runtime") and hc["Runtime"] not in ("runc", ""):
        add("--runtime", _q(hc["Runtime"]))
    if hc.get("IpcMode") == "host":
        add("--ipc", "host")
    elif hc.get("ShmSize") and hc["ShmSize"] != DEFAULT_SHM:
        add("--shm-size", _size_flag(hc["ShmSize"]))
    if hc.get("Memory"):
        add("--memory", _size_flag(hc["Memory"]))
    if hc.get("NanoCpus"):
        add("--cpus", f"{hc['NanoCpus'] / 1e9:g}")
    if hc.get("CpusetCpus"):
        add("--cpuset-cpus", _q(hc["CpusetCpus"]))
    if hc.get("NetworkMode") and hc["NetworkMode"] not in ("default", "bridge") and not hc["NetworkMode"].startswith("container:"):
        add("--network", _q(hc["NetworkMode"]))
    for ulimit in hc.get("Ulimits") or []:
        add("--ulimit", f"{ulimit['Name']}={ulimit['Soft']}:{ulimit['Hard']}")
    policy = (hc.get("RestartPolicy") or {}).get("Name")
    if policy and policy != "no":
        add("--restart", policy)
    for private, bindings in sorted((hc.get("PortBindings") or {}).items()):
        for b in bindings or []:
            host = f"{b['HostIp']}:" if b.get("HostIp") else ""
            port = private.removesuffix("/tcp")
            add("-p", _q(f"{host}{b.get('HostPort') or ''}:{port}"))
    for m in attrs.get("Mounts") or []:
        src = m.get("Name") if m.get("Type") == "volume" else m.get("Source")
        if not src or not m.get("Destination"):
            continue
        suffix = "" if m.get("RW", True) else ":ro"
        add("-v", _q(f"{src}:{m['Destination']}{suffix}"))
    baked = set(image_env or [])
    for item in cfg.get("Env") or []:
        if item in baked:
            continue
        key = item.partition("=")[0]
        if SECRET_KEY_RE.search(key):
            # Never copy secrets into drafts: reference the launching shell's env instead.
            add("-e", f'{key}="${{{key}}}"')
        else:
            add("-e", _q(item))
    if cfg.get("Entrypoint"):
        ep = cfg["Entrypoint"]
        add("--entrypoint", _q(ep[0]))
        extra_ep = list(ep[1:])
    else:
        extra_ep = []
    add(_q(cfg.get("Image", "")))
    args = extra_ep + list(cfg.get("Cmd") or [])
    i = 0
    while i < len(args):
        # Keep "--flag value" pairs on one line for readability.
        if args[i].startswith("-") and i + 1 < len(args) and not args[i + 1].startswith("-"):
            add(_q(args[i]), _q(args[i + 1]))
            i += 2
        else:
            add(_q(args[i]))
            i += 1
    return " \\\n  ".join(lines) + "\n"
