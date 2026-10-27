"""Resource analysis: engine flags, container limits, host GPU state and sanity checks.

The same analysis runs on a live container (from ``docker inspect``) and on a
draft (from a ``docker run`` command or a compose service), so drafts can be
linted before they are launched.
"""

from __future__ import annotations

import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

GIB = 1024**3
MIB = 1024**2

# --------------------------------------------------------------------------------
# Size / flag parsing helpers
# --------------------------------------------------------------------------------

_SIZE_RE = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*([kmgtp]?)i?b?\s*$", re.IGNORECASE)
_SIZE_MULT = {"": 1, "k": 1024, "m": MIB, "g": GIB, "t": 1024**4, "p": 1024**5}


def parse_size(value: Any) -> int | None:
    """Parse docker/compose sizes like ``16g``, ``512MB``, ``1.5GiB`` or raw bytes."""
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return int(value)
    m = _SIZE_RE.match(str(value))
    if not m:
        return None
    return int(float(m.group(1)) * _SIZE_MULT[m.group(2).lower()])


def fmt_bytes(n: int | float | None) -> str:
    if n is None:
        return "?"
    if n >= GIB:
        return f"{n / GIB:.1f} GiB"
    return f"{n / MIB:.0f} MiB"


def _norm_flag(token: str) -> str:
    return token.lstrip("-").replace("_", "-").lower()


def split_command(text: str) -> list[str]:
    """Shell-split a command, honouring ``\\`` line continuations and ``#`` comments."""
    text = re.sub(r"\\\r?\n", " ", text)
    lexer = shlex.shlex(text, posix=True)
    lexer.whitespace_split = True
    lexer.commenters = "#"
    try:
        return list(lexer)
    except ValueError:
        # Unbalanced quotes while the user is still typing: best effort.
        return text.split()


def expand_shell_wrappers(argv: list[str]) -> list[str]:
    """Unwrap ``sh -c "python -m sglang.launch_server ..."`` style commands."""
    out: list[str] = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok == "-c" and i > 0 and argv[i - 1].split("/")[-1] in ("sh", "bash", "zsh", "dash"):
            if i + 1 < len(argv):
                out.extend(split_command(argv[i + 1]))
                i += 2
                continue
        out.append(tok)
        i += 1
    return out


def parse_flags(argv: list[str]) -> tuple[dict[str, str | bool], list[str]]:
    """Generic ``--key value`` / ``--key=value`` / ``--switch`` parser."""
    flags: dict[str, str | bool] = {}
    positionals: list[str] = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok.startswith("-") and len(tok) > 1 and not _is_number(tok):
            if "=" in tok:
                key, val = tok.split("=", 1)
                flags[_norm_flag(key)] = val
            elif i + 1 < len(argv) and (not argv[i + 1].startswith("-") or _is_number(argv[i + 1])):
                flags[_norm_flag(tok)] = argv[i + 1]
                i += 1
            else:
                flags[_norm_flag(tok)] = True
        else:
            positionals.append(tok)
        i += 1
    return flags, positionals


def _is_number(tok: str) -> bool:
    try:
        float(tok)
        return True
    except ValueError:
        return False


def _first(flags: dict[str, Any], *names: str) -> Any:
    for n in names:
        if n in flags:
            return flags[n]
    return None


def _as_float(v: Any) -> float | None:
    try:
        return float(v) if v not in (None, True) else None
    except (TypeError, ValueError):
        return None


def _as_int(v: Any) -> int | None:
    f = _as_float(v)
    return int(f) if f is not None else None


# --------------------------------------------------------------------------------
# Engine configuration
# --------------------------------------------------------------------------------


@dataclass
class EngineConfig:
    engine: str | None = None  # "vllm" | "sglang" | None
    model: str | None = None
    gpu_fraction: float | None = None
    fraction_flag: str | None = None
    fraction_is_default: bool = False
    tp: int = 1
    pp: int = 1
    dp: int = 1
    max_len: int | None = None
    kv_cache_dtype: str | None = None
    swap_space_gib: float | None = None
    cpu_offload_gib: float | None = None
    max_running_requests: int | None = None
    flags: dict[str, str | bool] = field(default_factory=dict)

    @property
    def gpus_needed(self) -> int:
        return max(1, self.tp) * max(1, self.pp) * max(1, self.dp)


def parse_engine_args(argv: list[str], hint: str = "") -> EngineConfig:
    argv = expand_shell_wrappers(list(argv))
    joined = " ".join(argv + [hint]).lower()
    engine = "sglang" if "sglang" in joined else ("vllm" if "vllm" in joined else None)
    flags, positionals = parse_flags(argv)
    cfg = EngineConfig(engine=engine, flags=flags)

    if engine == "sglang":
        cfg.model = _first(flags, "model-path", "model")
        cfg.fraction_flag = "--mem-fraction-static"
        cfg.gpu_fraction = _as_float(flags.get("mem-fraction-static"))
        cfg.tp = _as_int(_first(flags, "tp-size", "tp", "tensor-parallel-size")) or 1
        cfg.dp = _as_int(_first(flags, "dp-size", "dp", "data-parallel-size")) or 1
        cfg.pp = _as_int(_first(flags, "pp-size", "pipeline-parallel-size")) or 1
        cfg.max_len = _as_int(flags.get("context-length"))
        cfg.max_running_requests = _as_int(flags.get("max-running-requests"))
        cfg.kv_cache_dtype = _first(flags, "kv-cache-dtype")
    elif engine == "vllm":
        model = _first(flags, "model")
        if model is None and "serve" in positionals:
            idx = positionals.index("serve")
            if idx + 1 < len(positionals):
                model = positionals[idx + 1]
        cfg.model = model
        cfg.fraction_flag = "--gpu-memory-utilization"
        frac = _as_float(flags.get("gpu-memory-utilization"))
        if frac is None:
            frac, cfg.fraction_is_default = 0.9, True
        cfg.gpu_fraction = frac
        cfg.tp = _as_int(_first(flags, "tensor-parallel-size", "tp")) or 1
        cfg.pp = _as_int(_first(flags, "pipeline-parallel-size", "pp")) or 1
        cfg.dp = _as_int(_first(flags, "data-parallel-size", "dp")) or 1
        cfg.max_len = _as_int(flags.get("max-model-len"))
        cfg.max_running_requests = _as_int(flags.get("max-num-seqs"))
        cfg.kv_cache_dtype = _first(flags, "kv-cache-dtype")
        cfg.swap_space_gib = _as_float(flags.get("swap-space"))
        cfg.cpu_offload_gib = _as_float(flags.get("cpu-offload-gb"))
    return cfg


# --------------------------------------------------------------------------------
# Container limits
# --------------------------------------------------------------------------------


@dataclass
class GpuAllotment:
    all: bool = False
    ids: list[str] | None = None  # indices or UUIDs
    count: int | None = None

    @property
    def requested(self) -> bool:
        return self.all or bool(self.ids) or bool(self.count)

    def describe(self) -> str:
        if self.all:
            return "all"
        if self.ids:
            return ",".join(self.ids)
        if self.count:
            return f"{self.count} (any)"
        return "none"


@dataclass
class ContainerLimits:
    memory: int | None = None  # bytes, None = unlimited
    memory_swap: int | None = None
    cpus: float | None = None
    cpuset: str | None = None
    shm_size: int | None = None
    ipc_mode: str | None = None
    gpus: GpuAllotment = field(default_factory=GpuAllotment)
    runtime: str | None = None
    env: dict[str, str] = field(default_factory=dict)

    @property
    def shm_effective(self) -> int:
        return self.shm_size or 64 * MIB  # docker default


def _env_to_dict(env: Iterable[str] | dict[str, Any] | None) -> dict[str, str]:
    if not env:
        return {}
    if isinstance(env, dict):
        return {str(k): "" if v is None else str(v) for k, v in env.items()}
    out = {}
    for item in env:
        k, _, v = str(item).partition("=")
        out[k] = v
    return out


def _gpus_from_visible_devices(env: dict[str, str]) -> GpuAllotment | None:
    val = env.get("NVIDIA_VISIBLE_DEVICES")
    if val is None:
        return None
    val = val.strip()
    if val in ("", "void", "none"):
        return GpuAllotment()
    if val == "all":
        return GpuAllotment(all=True)
    return GpuAllotment(ids=[v.strip() for v in val.split(",") if v.strip()])


def parse_gpus_flag(value: str) -> GpuAllotment:
    """Parse ``docker run --gpus`` values: ``all``, ``2``, ``device=0,1``, ``"device=0,1"``."""
    value = value.strip().strip('"').strip("'")
    if value == "all":
        return GpuAllotment(all=True)
    if value.isdigit():
        return GpuAllotment(count=int(value))
    m = re.search(r"device=([^\s]+)", value)
    if m:
        ids = [x.strip().strip('"') for x in m.group(1).split(",") if x.strip()]
        return GpuAllotment(ids=ids)
    m = re.search(r"count=(\w+)", value)
    if m:
        return GpuAllotment(all=True) if m.group(1) == "all" else GpuAllotment(count=_as_int(m.group(1)))
    return GpuAllotment()


def limits_from_inspect(attrs: dict[str, Any]) -> ContainerLimits:
    hc = attrs.get("HostConfig") or {}
    cfg = attrs.get("Config") or {}
    env = _env_to_dict(cfg.get("Env"))
    lim = ContainerLimits(
        memory=hc.get("Memory") or None,
        memory_swap=hc.get("MemorySwap") or None,
        cpuset=hc.get("CpusetCpus") or None,
        shm_size=hc.get("ShmSize") or None,
        ipc_mode=hc.get("IpcMode") or None,
        runtime=hc.get("Runtime") or None,
        env=env,
    )
    if hc.get("NanoCpus"):
        lim.cpus = hc["NanoCpus"] / 1e9
    elif hc.get("CpuQuota") and hc.get("CpuPeriod"):
        lim.cpus = hc["CpuQuota"] / hc["CpuPeriod"]
    gpus = GpuAllotment()
    for req in hc.get("DeviceRequests") or []:
        caps = [c for group in (req.get("Capabilities") or []) for c in group]
        if req.get("Driver") not in (None, "", "nvidia") and "gpu" not in caps:
            continue
        if req.get("DeviceIDs"):
            gpus = GpuAllotment(ids=list(req["DeviceIDs"]))
        elif req.get("Count") == -1:
            gpus = GpuAllotment(all=True)
        elif req.get("Count"):
            gpus = GpuAllotment(count=req["Count"])
    if not gpus.requested and lim.runtime == "nvidia":
        gpus = _gpus_from_visible_devices(env) or GpuAllotment()
    lim.gpus = gpus
    return lim


def argv_from_inspect(attrs: dict[str, Any]) -> list[str]:
    cfg = attrs.get("Config") or {}
    return list(cfg.get("Entrypoint") or []) + list(cfg.get("Cmd") or [])


# --------------------------------------------------------------------------------
# docker run / compose parsing (for drafts)
# --------------------------------------------------------------------------------

DOCKER_RUN_BOOL_FLAGS = {
    "d", "detach", "i", "interactive", "t", "tty", "rm", "privileged", "init", "P",
    "publish-all", "read-only", "oom-kill-disable", "no-healthcheck", "sig-proxy", "q", "quiet",
}


@dataclass
class RunSpec:
    """A parsed ``docker run`` invocation."""

    image: str | None = None
    name: str | None = None
    argv: list[str] = field(default_factory=list)  # command inside the container
    limits: ContainerLimits = field(default_factory=ContainerLimits)
    ports: list[str] = field(default_factory=list)
    volumes: list[str] = field(default_factory=list)


def parse_docker_run(text: str) -> RunSpec | None:
    tokens = split_command(text)
    # find "docker run" / "docker container run"
    start = None
    for i, tok in enumerate(tokens):
        if tok.split("/")[-1] in ("docker", "podman", "nerdctl"):
            rest = tokens[i + 1 : i + 3]
            if rest[:1] == ["run"]:
                start = i + 2
            elif rest == ["container", "run"]:
                start = i + 3
            if start is not None:
                break
    if start is None:
        return None
    spec = RunSpec()
    lim = spec.limits
    env: dict[str, str] = {}
    i = start
    while i < len(tokens):
        tok = tokens[i]
        if tok in ("&&", "||", ";", "|"):
            break
        if not tok.startswith("-") or tok == "-":
            spec.image = tok
            rest = []
            for t in tokens[i + 1 :]:
                if t in ("&&", "||", ";", "|"):
                    break
                rest.append(t)
            spec.argv = rest
            break
        if "=" in tok and tok.startswith("--"):
            key, val = tok[2:].split("=", 1)
        else:
            key = tok.lstrip("-")
            if (tok.startswith("--") and key in DOCKER_RUN_BOOL_FLAGS) or (
                not tok.startswith("--") and all(c in DOCKER_RUN_BOOL_FLAGS for c in key)
            ):
                i += 1
                continue
            val = tokens[i + 1] if i + 1 < len(tokens) else ""
            i += 1
        i += 1
        if key == "gpus":
            lim.gpus = parse_gpus_flag(val)
        elif key in ("m", "memory"):
            lim.memory = parse_size(val)
        elif key == "memory-swap":
            lim.memory_swap = parse_size(val)
        elif key == "cpus":
            lim.cpus = _as_float(val)
        elif key == "cpuset-cpus":
            lim.cpuset = val
        elif key == "shm-size":
            lim.shm_size = parse_size(val)
        elif key == "ipc":
            lim.ipc_mode = val
        elif key == "runtime":
            lim.runtime = val
        elif key in ("e", "env"):
            k, _, v = val.partition("=")
            env[k] = v
        elif key == "name":
            spec.name = val
        elif key in ("p", "publish"):
            spec.ports.append(val)
        elif key in ("v", "volume", "mount"):
            spec.volumes.append(val)
    lim.env = env
    if not lim.gpus.requested and lim.runtime == "nvidia":
        lim.gpus = _gpus_from_visible_devices(env) or GpuAllotment()
    return spec


def _compose_argv(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    return split_command(str(value))


def spec_from_compose_service(name: str, svc: dict[str, Any]) -> RunSpec:
    spec = RunSpec(image=svc.get("image"), name=svc.get("container_name") or name)
    spec.argv = _compose_argv(svc.get("entrypoint")) + _compose_argv(svc.get("command"))
    lim = spec.limits
    lim.env = _env_to_dict(svc.get("environment"))
    lim.shm_size = parse_size(svc.get("shm_size"))
    lim.ipc_mode = svc.get("ipc")
    lim.runtime = svc.get("runtime")
    lim.memory = parse_size(svc.get("mem_limit"))
    lim.cpus = _as_float(svc.get("cpus"))
    lim.cpuset = svc.get("cpuset")
    resources = ((svc.get("deploy") or {}).get("resources")) or {}
    limits = resources.get("limits") or {}
    if limits.get("memory"):
        lim.memory = parse_size(limits["memory"])
    if limits.get("cpus"):
        lim.cpus = _as_float(limits["cpus"])
    for dev in (resources.get("reservations") or {}).get("devices") or []:
        caps = dev.get("capabilities") or []
        if dev.get("driver") not in (None, "nvidia") and "gpu" not in caps:
            continue
        if dev.get("device_ids"):
            lim.gpus = GpuAllotment(ids=[str(d) for d in dev["device_ids"]])
        elif str(dev.get("count", "")).lower() == "all":
            lim.gpus = GpuAllotment(all=True)
        elif dev.get("count"):
            lim.gpus = GpuAllotment(count=_as_int(dev["count"]))
        else:
            lim.gpus = GpuAllotment(all=True)  # compose default when count/ids omitted
    if not lim.gpus.requested and svc.get("gpus") is not None:
        g = svc["gpus"]
        lim.gpus = GpuAllotment(all=True) if g == "all" else GpuAllotment(count=_as_int(g))
    if not lim.gpus.requested and lim.runtime == "nvidia":
        lim.gpus = _gpus_from_visible_devices(lim.env) or GpuAllotment()
    for p in svc.get("ports") or []:
        spec.ports.append(str(p) if not isinstance(p, dict) else f"{p.get('published')}:{p.get('target')}")
    for v in svc.get("volumes") or []:
        spec.volumes.append(str(v) if not isinstance(v, dict) else f"{v.get('source')}:{v.get('target')}")
    return spec


# --------------------------------------------------------------------------------
# nvidia-smi
# --------------------------------------------------------------------------------


@dataclass
class GpuInfo:
    index: int
    uuid: str
    name: str
    total_mib: int
    used_mib: int
    util_pct: int | None = None

    @property
    def free_mib(self) -> int:
        return max(0, self.total_mib - self.used_mib)


@dataclass
class ComputeApp:
    pid: int
    gpu_uuid: str
    used_mib: int


@dataclass
class GpuSnapshot:
    gpus: list[GpuInfo] = field(default_factory=list)
    apps: list[ComputeApp] = field(default_factory=list)
    error: str | None = None

    @property
    def available(self) -> bool:
        return bool(self.gpus)

    def resolve(self, allot: GpuAllotment) -> list[GpuInfo]:
        """Map an allotment (indices, UUIDs, count, all) onto host GPUs."""
        if allot.all:
            return list(self.gpus)
        if allot.ids:
            out = []
            for ident in allot.ids:
                for g in self.gpus:
                    if ident == str(g.index) or ident == g.uuid or (len(ident) > 6 and g.uuid.startswith(ident)):
                        out.append(g)
                        break
            return out
        if allot.count:
            return list(self.gpus[: allot.count])
        return []

    def usage_by_pids(self, pids: set[int]) -> dict[str, int]:
        """MiB used per GPU UUID by the given host PIDs."""
        out: dict[str, int] = {}
        for app in self.apps:
            if app.pid in pids:
                out[app.gpu_uuid] = out.get(app.gpu_uuid, 0) + app.used_mib
        return out


def _csv_rows(text: str) -> list[list[str]]:
    rows = []
    for line in text.strip().splitlines():
        if line.strip():
            rows.append([c.strip() for c in line.split(",")])
    return rows


def parse_gpu_csv(text: str) -> list[GpuInfo]:
    gpus = []
    for row in _csv_rows(text):
        if len(row) < 5:
            continue
        try:
            gpus.append(
                GpuInfo(
                    index=int(row[0]),
                    uuid=row[1],
                    name=row[2],
                    total_mib=int(float(row[3])),
                    used_mib=int(float(row[4])),
                    util_pct=_as_int(row[5]) if len(row) > 5 else None,
                )
            )
        except ValueError:
            continue
    return gpus


def parse_apps_csv(text: str) -> list[ComputeApp]:
    apps = []
    for row in _csv_rows(text):
        if len(row) < 3:
            continue
        try:
            apps.append(ComputeApp(pid=int(row[0]), gpu_uuid=row[1], used_mib=int(float(row[2]))))
        except ValueError:
            continue
    return apps


Runner = Callable[[list[str]], str]


def _default_runner(cmd: list[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=5).stdout


def query_gpus(runner: Runner | None = None) -> GpuSnapshot:
    if runner is None:
        if shutil.which("nvidia-smi") is None:
            return GpuSnapshot(error="nvidia-smi not found")
        runner = _default_runner
    try:
        gpus = parse_gpu_csv(
            runner(
                [
                    "nvidia-smi",
                    "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ]
            )
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return GpuSnapshot(error=f"nvidia-smi failed: {exc}")
    try:
        apps = parse_apps_csv(
            runner(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,gpu_uuid,used_memory",
                    "--format=csv,noheader,nounits",
                ]
            )
        )
    except (OSError, subprocess.SubprocessError):
        apps = []
    return GpuSnapshot(gpus=gpus, apps=apps)


# --------------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------------

SEVERITY_ICON = {"ok": "✅", "info": "ℹ️ ", "warn": "⚠️ ", "error": "❌"}


@dataclass
class Finding:
    severity: str  # ok | info | warn | error
    message: str

    def __str__(self) -> str:
        return f"{SEVERITY_ICON.get(self.severity, '')} {self.message}"


@dataclass
class Peer:
    """Another container that may share GPUs with the one being analysed."""

    name: str
    gpus: GpuAllotment
    fraction: float | None


def analyze(
    engine: EngineConfig,
    limits: ContainerLimits,
    snapshot: GpuSnapshot | None = None,
    *,
    running: bool = False,
    own_usage_mib: dict[str, int] | None = None,
    peers: Iterable[Peer] = (),
) -> list[Finding]:
    findings: list[Finding] = []
    snapshot = snapshot or GpuSnapshot()
    own_usage_mib = own_usage_mib or {}

    if engine.engine is None:
        findings.append(Finding("info", "Could not detect vLLM/SGLang command line; engine checks skipped."))

    # --- GPU allotment vs parallelism ---
    if not limits.gpus.requested:
        findings.append(Finding("error", "No GPUs requested (--gpus / deploy.resources.reservations.devices)."))
    else:
        if limits.gpus.all:
            allotted = len(snapshot.gpus) if snapshot.available else None
        elif limits.gpus.ids:
            allotted = len(limits.gpus.ids)
        else:
            allotted = limits.gpus.count
        need = engine.gpus_needed
        if engine.engine and allotted is not None and need > allotted:
            findings.append(
                Finding("error", f"Parallelism needs {need} GPUs (tp={engine.tp} pp={engine.pp} dp={engine.dp}) "
                        f"but only {allotted} allotted ({limits.gpus.describe()}).")
            )
        elif engine.engine and allotted is not None and need < allotted and not limits.gpus.all:
            findings.append(
                Finding("info", f"{allotted} GPUs allotted but only {need} used by tp/pp/dp.")
            )

    # --- memory fraction ---
    frac = engine.gpu_fraction
    if engine.engine == "sglang" and frac is None:
        findings.append(Finding("info", "--mem-fraction-static not set: SGLang picks it automatically."))
    if frac is not None:
        label = f"{engine.fraction_flag}={frac:g}" + (" (default)" if engine.fraction_is_default else "")
        if frac > 1.0 or frac <= 0:
            findings.append(Finding("error", f"{label} is outside (0, 1]."))
        elif frac > 0.95:
            findings.append(Finding("warn", f"{label} leaves <5% for CUDA graphs/activations; OOM risk."))

    # --- per-GPU budget vs current usage ---
    if snapshot.available and limits.gpus.requested:
        targets = snapshot.resolve(limits.gpus)
        if engine.engine:
            targets = targets[: engine.gpus_needed] if not limits.gpus.ids else targets
        for g in targets:
            own = own_usage_mib.get(g.uuid, 0)
            others = max(0, g.used_mib - own)
            if frac is not None:
                budget = frac * g.total_mib
                free_for_us = g.total_mib - others
                if budget > free_for_us:
                    verb = "needs" if not running else "reserves"
                    findings.append(
                        Finding("error", f"GPU {g.index} ({g.name}): {verb} {budget / 1024:.1f} GiB "
                                f"({frac:g}×{g.total_mib / 1024:.0f} GiB) but other processes use "
                                f"{others / 1024:.1f} GiB → only {free_for_us / 1024:.1f} GiB available.")
                    )
                else:
                    findings.append(
                        Finding("ok", f"GPU {g.index} ({g.name}): budget {budget / 1024:.1f} GiB, "
                                f"{free_for_us / 1024:.1f} GiB available"
                                + (f", this container uses {own / 1024:.1f} GiB" if own else "") + ".")
                    )
            elif others and not running:
                findings.append(Finding("info", f"GPU {g.index}: {others / 1024:.1f} GiB already in use by other processes."))
    elif limits.gpus.requested and snapshot.error:
        findings.append(Finding("info", f"Live GPU check skipped: {snapshot.error}."))

    # --- peers sharing GPUs ---
    if frac is not None and limits.gpus.requested:
        def gpu_set(allot: GpuAllotment) -> set[str]:
            if snapshot.available:
                return {g.uuid for g in snapshot.resolve(allot)}
            return set(allot.ids) if allot.ids else {"*"}  # unknown placement: assume overlap

        mine = gpu_set(limits.gpus)
        for peer in peers:
            if peer.fraction is None or not peer.gpus.requested:
                continue
            theirs = gpu_set(peer.gpus)
            shared = bool(mine & theirs) or "*" in mine or "*" in theirs
            if shared and frac + peer.fraction > 1.0:
                findings.append(
                    Finding("error", f"Shares GPU(s) with running '{peer.name}' ({peer.fraction:g}); "
                            f"fractions sum to {frac + peer.fraction:.2f} > 1.0.")
                )

    # --- shared memory ---
    if limits.ipc_mode != "host" and limits.shm_effective < GIB:
        sev = "error" if engine.gpus_needed > 1 else "warn"
        findings.append(
            Finding(sev, f"shm is {fmt_bytes(limits.shm_effective)} without --ipc=host; "
                    "use --ipc=host or --shm-size≥8g (NCCL/torch workers need it).")
        )

    # --- host RAM / CPU ---
    host_ram_needed = (engine.swap_space_gib or 0) + (engine.cpu_offload_gib or 0)
    if limits.memory:
        if host_ram_needed and host_ram_needed * GIB >= limits.memory:
            findings.append(
                Finding("error", f"RAM limit {fmt_bytes(limits.memory)} ≤ swap-space+cpu-offload "
                        f"({host_ram_needed:g} GiB).")
            )
        elif limits.memory < 16 * GIB:
            findings.append(Finding("warn", f"RAM limit {fmt_bytes(limits.memory)}: model loading may be OOM-killed."))
    if limits.cpus is not None and limits.cpus < 2:
        findings.append(Finding("warn", f"CPU limit {limits.cpus:g} cores: tokenization/scheduling may bottleneck."))

    return findings


def summarize_limits(limits: ContainerLimits) -> list[tuple[str, str]]:
    rows = [
        ("GPUs", limits.gpus.describe()),
        ("Memory", fmt_bytes(limits.memory) if limits.memory else "unlimited"),
        ("CPUs", f"{limits.cpus:g}" if limits.cpus else "unlimited"),
    ]
    if limits.cpuset:
        rows.append(("CPU set", limits.cpuset))
    rows.append(("Shared mem", "host IPC" if limits.ipc_mode == "host" else fmt_bytes(limits.shm_effective)))
    if limits.runtime:
        rows.append(("Runtime", limits.runtime))
    return rows


def summarize_engine(engine: EngineConfig) -> list[tuple[str, str]]:
    if engine.engine is None:
        return [("Engine", "unknown")]
    rows = [("Engine", engine.engine), ("Model", engine.model or "?")]
    if engine.gpu_fraction is not None:
        rows.append(
            (engine.fraction_flag or "fraction", f"{engine.gpu_fraction:g}" + (" (default)" if engine.fraction_is_default else ""))
        )
    else:
        rows.append((engine.fraction_flag or "fraction", "auto"))
    rows.append(("Parallelism", f"tp={engine.tp} pp={engine.pp} dp={engine.dp}"))
    if engine.max_len:
        rows.append(("Max context", str(engine.max_len)))
    if engine.kv_cache_dtype:
        rows.append(("KV cache dtype", str(engine.kv_cache_dtype)))
    if engine.max_running_requests:
        rows.append(("Max running reqs", str(engine.max_running_requests)))
    if engine.swap_space_gib is not None:
        rows.append(("Swap space", f"{engine.swap_space_gib:g} GiB"))
    if engine.cpu_offload_gib is not None:
        rows.append(("CPU offload", f"{engine.cpu_offload_gib:g} GiB"))
    return rows
