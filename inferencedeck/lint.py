"""Lint drafts and live containers with the resource analyzer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from .compose import load_compose_services
from .resources import (
    ContainerLimits,
    EngineConfig,
    Finding,
    GpuSnapshot,
    Peer,
    analyze,
    argv_from_inspect,
    limits_from_inspect,
    parse_docker_run,
    parse_engine_args,
    spec_from_compose_service,
)


@dataclass
class Report:
    """Analysis of one container or one draft service."""

    scope: str
    engine: EngineConfig
    limits: ContainerLimits
    findings: list[Finding] = field(default_factory=list)


def count_by_severity(reports: Iterable[Report]) -> dict[str, int]:
    """Count errors and warnings across reports."""
    counts = {"error": 0, "warn": 0}
    for r in reports:
        for f in r.findings:
            if f.severity in counts:
                counts[f.severity] += 1
    return counts


def report_for_container(
    attrs: dict[str, Any],
    snapshot: GpuSnapshot | None,
    *,
    own_usage_mib: dict[str, int] | None = None,
    peers: Iterable[Peer] = (),
) -> Report:
    """Analyse a live container from its inspect payload."""
    name = (attrs.get("Name") or "").lstrip("/")
    image = (attrs.get("Config") or {}).get("Image", "")
    engine = parse_engine_args(argv_from_inspect(attrs), hint=image)
    limits = limits_from_inspect(attrs)
    running = ((attrs.get("State") or {}).get("Status")) == "running"
    findings = analyze(
        engine,
        limits,
        snapshot,
        running=running,
        own_usage_mib=own_usage_mib,
        peers=[p for p in peers if p.name != name],
    )
    return Report(scope=name, engine=engine, limits=limits, findings=findings)


def peer_from_inspect(attrs: dict[str, Any]) -> Peer:
    """Summarise a running container's GPUs and memory fraction for overlap checks."""
    image = (attrs.get("Config") or {}).get("Image", "")
    engine = parse_engine_args(argv_from_inspect(attrs), hint=image)
    return Peer(
        name=(attrs.get("Name") or "").lstrip("/"),
        gpus=limits_from_inspect(attrs).gpus,
        fraction=engine.gpu_fraction,
    )


def lint_draft(
    kind: str,
    body: str,
    snapshot: GpuSnapshot | None = None,
    *,
    peers: Iterable[Peer] = (),
    existing_names: Iterable[str] = (),
) -> list[Report]:
    """Analyse a draft (one report for a run draft, one per vLLM/SGLang service for compose)."""
    peers = list(peers)
    existing = set(existing_names)
    if kind == "run":
        spec = parse_docker_run(body)
        if spec is None:
            return [Report("draft", EngineConfig(), ContainerLimits(), [Finding("warn", "No 'docker run' command found.")])]
        engine = parse_engine_args(spec.argv, hint=spec.image or "")
        others = [p for p in peers if p.name != spec.name]
        findings = analyze(engine, spec.limits, snapshot, peers=others)
        if not spec.image:
            findings.insert(0, Finding("error", "No image given."))
        if spec.name and spec.name in existing:
            findings.insert(
                0, Finding("error", f"A container named '{spec.name}' already exists; remove it or rename the draft's --name.")
            )
        return [Report(spec.name or spec.image or "draft", engine, spec.limits, findings)]

    if kind == "compose":
        try:
            services = load_compose_services(body)
        except ValueError as exc:
            return [Report("compose", EngineConfig(), ContainerLimits(), [Finding("error", str(exc))])]
        if not services:
            return [Report("compose", EngineConfig(), ContainerLimits(), [Finding("warn", "No services defined.")])]
        reports = []
        for svc_name, svc in services.items():
            spec = spec_from_compose_service(svc_name, svc)
            engine = parse_engine_args(spec.argv, hint=spec.image or "")
            if engine.engine is None and not spec.limits.gpus.requested:
                continue  # sidecars (proxies, UIs...) are not analysed
            # Compose recreates its own containers, so a same-named peer is the old copy of this service.
            others = [p for p in peers if p.name != spec.name]
            findings = analyze(engine, spec.limits, snapshot, peers=others)
            if not spec.image and not svc.get("build"):
                findings.insert(0, Finding("error", "No image or build given."))
            reports.append(Report(f"service {svc_name}", engine, spec.limits, findings))
        if not reports:
            reports.append(Report("compose", EngineConfig(), ContainerLimits(), [Finding("info", "No vLLM/SGLang services found.")]))
        return reports

    raise ValueError(f"unknown draft kind {kind!r}")
