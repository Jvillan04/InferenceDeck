"""InferenceDeck main screen: discover, control and inspect vLLM / SGLang containers."""

from __future__ import annotations

import asyncio
import dataclasses
import functools
import os
import threading
import time
from datetime import datetime
from typing import Any, Callable, Iterable

from docker.errors import DockerException, NotFound
from rich.markup import escape
from rich.table import Table
from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult, SystemCommand
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.screen import Screen
from textual.widgets import (
    Footer,
    Header,
    Label,
    ListItem,
    ListView,
    Log,
    Static,
    TabbedContent,
    TabPane,
    TextArea,
)
from textual.worker import get_current_worker

from . import updater
from .backend import ContainerInfo, DockerBackend, DockerUnavailable
from .compose import compose_info_from_labels, reconstruct_docker_run
from .drafts import DraftStore
from .lint import Report, peer_from_inspect, report_for_container
from .resources import GpuSnapshot, Peer, query_gpus, summarize_engine, summarize_limits
from .screens import (
    SEVERITY_STYLE,
    ConfirmScreen,
    DraftFormScreen,
    DraftsScreen,
    UpdateScreen,
    set_language,
)

PENDING_LABEL = {"start": "starting…", "stop": "stopping…", "restart": "restarting…"}
UPDATE_CHECK_INTERVAL = 24 * 3600

CSS = """
Screen { background: $surface; }
#sidebar, #drafts-sidebar { width: 30%; border-right: solid $primary; background: $surface-darken-1; }
.sidebar-title { text-style: bold; padding: 0 1; width: 100%; background: $primary-darken-2; }
.hint { color: $text-muted; padding: 0 1; }
#main-tabs, #drafts-main { width: 70%; }
#drafts-main { padding: 0 1; }
#details, #resources { padding: 0 1; }
#log-output, #launch-log { background: $background; height: 1fr; border: tall $background; }
#compose-meta { padding: 0 1; height: auto; }
#compose-text { height: 1fr; }
#draft-title { padding: 0 1; height: auto; max-height: 3; width: 100%; }
#editor { height: 2fr; }
#lint-panel { height: 1fr; max-height: 14; border: round $primary-darken-2; }
#launch-log { height: 8; }
ContainerItem.-pending Label { color: $warning; }
ModalScreen { align: center middle; }
.dialog { width: 70; height: auto; max-height: 90%; border: thick $primary; background: $panel; padding: 1 2; }
.dialog.wide { width: 90%; height: 85%; }
.dialog-title { text-style: bold; margin-bottom: 1; }
.dialog-body { margin-bottom: 1; }
.dialog-buttons { height: auto; margin-top: 1; align-horizontal: right; }
.dialog-buttons Button { margin-left: 1; }
#rev-body { height: 1fr; }
#rev-list { width: 30; }
#update-log { height: 1fr; }
.hidden { display: none; }
"""


def ignore_if_unmounted(func):
    """Let a background task that finishes during shutdown exit quietly instead of failing on removed widgets."""
    if asyncio.iscoroutinefunction(func):

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except NoMatches:
                return None

        return async_wrapper

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except NoMatches:
            return None

    return wrapper


def short_image(image: str) -> str:
    """``docker.io/vllm/vllm-openai:v0.6.3`` -> ``vllm-openai:v0.6.3``; digests are shortened."""
    if image.startswith("sha256:"):
        return image[7:19]
    return image.rsplit("/", 1)[-1]


class ContainerItem(ListItem):
    """Sidebar row for one container: status emoji, name and short image."""
    def __init__(self, info: ContainerInfo, pending: str | None = None) -> None:
        self.container_id = info.id
        self.info = info
        # Hold the label directly: polls may update it before the child is mounted (query_one would fail).
        self.label = Label(self.render_label(info, pending))
        super().__init__(self.label)
        self.set_class(pending is not None, "-pending")

    @staticmethod
    def render_label(info: ContainerInfo, pending: str | None = None) -> str:
        """Label markup; a pending action replaces the emoji with ⏳."""
        status = f"⏳ {pending}" if pending else info.emoji
        return f"{status} [b]{escape(info.name)}[/]\n   [dim]{escape(short_image(info.image))}[/]"

    def set_info(self, info: ContainerInfo, pending: str | None = None) -> None:
        """Update the row in place with new state."""
        self.info = info
        self.label.update(self.render_label(info, pending))
        self.set_class(pending is not None, "-pending")


class InferenceDeckApp(App[str | None]):
    """Main screen: container list, Details/Logs/Resources/Compose tabs and lifecycle actions."""
    TITLE = "InferenceDeck"
    SUB_TITLE = "vLLM · SGLang"
    CSS = CSS

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("s", "start", "Start"),
        Binding("t", "stop", "Stop"),
        Binding("r", "restart", "Restart"),
        Binding("l", "attach_logs", "Logs"),
        Binding("d", "drafts", "Drafts"),
        Binding("e", "save_as_draft", "Save as draft"),
        Binding("ctrl+r,f5", "refresh_containers", "Refresh"),
        Binding("U", "check_updates", "Update"),
    ]

    def __init__(
        self,
        backend: DockerBackend | None = None,
        store: DraftStore | None = None,
        *,
        db_path: str | None = None,
        gpu_query: Callable[[], GpuSnapshot] = query_gpus,
        poll_interval: float = 2.0,
        gpu_interval: float = 5.0,
        check_updates: bool | None = None,
    ) -> None:
        super().__init__()
        self.backend = backend or DockerBackend()
        self.store = store or DraftStore(db_path)
        self.gpu_query = gpu_query
        self.poll_interval = poll_interval
        self.gpu_interval = gpu_interval
        if check_updates is None:
            check_updates = not os.environ.get("INFERENCEDECK_NO_UPDATE_CHECK")
        self.check_updates_on_start = check_updates

        self._infos: dict[str, ContainerInfo] = {}
        self._pending: dict[str, str] = {}
        self._inspect_cache: dict[str, tuple[str, dict[str, Any]]] = {}
        self._scan_running = False
        self._docker_error: str | None = None
        self.gpu_snapshot: GpuSnapshot = GpuSnapshot(error="not queried yet")
        self.peers: list[Peer] = []

        self._log_lock = threading.Lock()
        self._log_stream: Any = None
        self._log_container_id: str | None = None

    # -- layout --------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal():
            with Vertical(id="sidebar"):
                yield Label("🤖 LLM Containers", classes="sidebar-title")
                yield ListView(id="container-list")
                yield Static("", id="sidebar-status", classes="hint")
            with TabbedContent(id="main-tabs", initial="tab-details"):
                with TabPane("Details", id="tab-details"):
                    yield VerticalScroll(Static("Select a container.", id="details"))
                with TabPane("Logs", id="tab-logs"):
                    yield Log(id="log-output", max_lines=5000, highlight=False)
                with TabPane("Resources", id="tab-resources"):
                    yield VerticalScroll(Static("Select a container.", id="resources"))
                with TabPane("Compose / Run", id="tab-compose"):
                    yield Static("Select a container.", id="compose-meta")
                    yield TextArea("", read_only=True, id="compose-text", show_line_numbers=True)
        yield Footer()

    def on_mount(self) -> None:
        """Start polling Docker and the GPUs, and optionally check for updates."""
        self.query_one("#container-list", ListView).focus()
        self.refresh_containers()
        self.refresh_host()
        self.set_interval(self.poll_interval, self.refresh_containers)
        self.set_interval(self.gpu_interval, self.refresh_host)
        if self.check_updates_on_start:
            self.quiet_update_check()

    def on_unmount(self) -> None:
        """Stop the log stream on exit."""
        self._close_log_stream()

    def get_system_commands(self, screen: Screen) -> Iterable[SystemCommand]:
        """Add InferenceDeck actions to the command palette."""
        yield from super().get_system_commands(screen)
        yield SystemCommand("Drafts", "Open the drafts editor", self.action_drafts)
        yield SystemCommand("Check for updates", "Pull the latest InferenceDeck", self.action_check_updates)
        yield SystemCommand("Refresh containers", "Rescan Docker now", self.action_refresh_containers)

    # -- helpers used by other screens ---------------------------------------------

    @property
    def container_names(self) -> list[str]:
        """Names of all listed containers (used to flag name clashes in drafts)."""
        return [i.name for i in self._infos.values()]

    @property
    def list_view(self) -> ListView:
        """The sidebar list widget."""
        return self.query_one("#container-list", ListView)

    def highlighted_info(self) -> ContainerInfo | None:
        """Latest state of the container under the cursor, or None."""
        item = self.list_view.highlighted_child
        if isinstance(item, ContainerItem):
            return self._infos.get(item.container_id, item.info)
        return None

    def _item_for(self, container_id: str) -> ContainerItem | None:
        """Sidebar row for a container ID, or None."""
        for item in self.list_view.query(ContainerItem):
            if item.container_id == container_id:
                return item
        return None

    def _set_status(self, text: str) -> None:
        """Set the status line under the sidebar."""
        self.query_one("#sidebar-status", Static).update(text)

    # -- discovery -----------------------------------------------------------------

    def action_refresh_containers(self) -> None:
        """Key binding: rescan Docker now."""
        self.refresh_containers()

    def refresh_containers(self) -> None:
        """Start a scan unless one is already running."""
        if not self._scan_running:  # never pile up scans behind a slow daemon
            self._scan_running = True
            self.run_worker(self._scan(), group="scan")

    @ignore_if_unmounted
    async def _scan(self) -> None:
        """List containers off the UI thread and apply the result; report Docker outages."""
        try:
            try:
                infos = await asyncio.to_thread(self.backend.list_llm_containers)
            except DockerUnavailable as exc:
                if self._docker_error is None:
                    self.notify(f"Docker unavailable: {exc}", severity="error", timeout=8)
                    self._docker_error = str(exc)
                    self._show_empty()
                self._set_status("[red]● Docker unavailable — retrying…[/]")
                return
            if self._docker_error is not None:
                self.notify("Reconnected to Docker.")
                self._docker_error = None
            await self._apply_scan(infos)
        finally:
            self._scan_running = False

    async def _apply_scan(self, infos: list[ContainerInfo]) -> None:
        """Update the sidebar, keeping the cursor on the same container."""
        lv = self.list_view
        before = self.highlighted_info()
        previous = self._infos
        self._infos = {i.id: i for i in infos}
        current_ids = [item.container_id for item in lv.query(ContainerItem)]
        new_ids = [i.id for i in infos]

        if current_ids == new_ids:
            # Same containers in the same order: update labels in place (no flicker, cursor untouched).
            for item in lv.query(ContainerItem):
                item.set_info(self._infos[item.container_id], self._pending.get(item.container_id))
        else:
            keep_id = before.id if before else None
            old_index = lv.index
            await lv.clear()
            await lv.extend(ContainerItem(i, self._pending.get(i.id)) for i in infos)
            # clear() reset the index to None; if it's set now, the user moved the cursor while we
            # were awaiting the remount, and that newer choice wins over our restore.
            if infos and lv.index is None:
                if keep_id in new_ids:
                    lv.index = new_ids.index(keep_id)
                else:
                    lv.index = min(old_index or 0, len(infos) - 1)

        for cid in list(self._inspect_cache):
            if cid not in self._infos:
                self._inspect_cache.pop(cid, None)

        running = sum(1 for i in infos if i.state == "running")
        self._set_status(f"{len(infos)} container(s) · {running} running · {datetime.now():%H:%M:%S}")

        after = self.highlighted_info()
        if after is None:
            if before is not None or not infos:
                self._show_empty()
        elif before is None or after.id != before.id or previous.get(after.id) != after:
            self.load_panels(after)

    def _show_empty(self) -> None:
        """Show placeholders when there are no containers or Docker is down."""
        msg = "No vLLM / SGLang containers found." if self._docker_error is None else "Docker unavailable."
        self.query_one("#details", Static).update(msg)
        self.query_one("#resources", Static).update(msg)
        self.query_one("#compose-meta", Static).update(msg)
        self.query_one("#compose-text", TextArea).load_text("")

    @on(ListView.Highlighted, "#container-list")
    def _on_highlight(self, event: ListView.Highlighted) -> None:
        """Cursor moved: load the tabs for that container."""
        if isinstance(event.item, ContainerItem):
            self.load_panels(self._infos.get(event.item.container_id, event.item.info))

    @on(ListView.Selected, "#container-list")
    def _on_select(self, event: ListView.Selected) -> None:
        """Enter pressed: tail the container's logs."""
        if isinstance(event.item, ContainerItem):
            self.attach_logs(event.item.container_id)
            self.query_one(TabbedContent).active = "tab-logs"

    # -- host / GPU state ----------------------------------------------------------

    def _inspect(self, info: ContainerInfo) -> dict[str, Any]:
        """Inspect with a cache keyed on state; Config/HostConfig don't change without a new ID."""
        cached = self._inspect_cache.get(info.id)
        if cached and cached[0] == info.state:
            return cached[1]
        attrs = self.backend.inspect(info.id)
        self._inspect_cache[info.id] = (info.state, attrs)
        return attrs

    @work(thread=True, exclusive=True, group="host")
    def refresh_host(self) -> None:
        """Refresh GPU usage and running containers' GPU claims in the background."""
        snapshot = self.gpu_query()
        peers = []
        for info in list(self._infos.values()):
            if info.state != "running":
                continue
            try:
                peers.append(peer_from_inspect(self._inspect(info)))
            except DockerException:
                continue
        self.gpu_snapshot = snapshot
        self.peers = peers

    # -- panels --------------------------------------------------------------------

    def load_panels(self, info: ContainerInfo) -> None:
        """Show basic details now and load the rest of the tabs in the background."""
        self.query_one("#details", Static).update(self._details_table(info, None))
        self._load_panels_worker(info)

    @work(thread=True, exclusive=True, group="panels")
    def _load_panels_worker(self, info: ContainerInfo) -> None:
        """Inspect a container and render its Details, Resources and Compose tabs."""
        worker = get_current_worker()
        try:
            attrs = self._inspect(info)
        except NotFound:
            return
        except DockerException as exc:
            self.call_from_thread(self.query_one("#resources", Static).update, f"Inspect failed: {exc}")
            return
        own_usage = {}
        if info.state == "running" and self.gpu_snapshot.apps:
            own_usage = self.gpu_snapshot.usage_by_pids(self.backend.top_pids(info.id))
        report = report_for_container(attrs, self.gpu_snapshot, own_usage_mib=own_usage, peers=self.peers)
        compose = compose_info_from_labels(info.labels)
        if compose is None or not compose.readable:
            run_cmd = reconstruct_docker_run(attrs, self.backend.image_env(info.image))
        else:
            run_cmd = None
        if worker.is_cancelled:
            return

        @ignore_if_unmounted
        def apply() -> None:
            current = self.highlighted_info()
            if current is None or current.id != info.id:
                return  # user moved on while we were loading
            self.query_one("#details", Static).update(self._details_table(info, attrs))
            self.query_one("#resources", Static).update(self._resources_view(report, own_usage))
            self._show_compose(info, compose, run_cmd)

        self.call_from_thread(apply)

    def _details_table(self, info: ContainerInfo, attrs: dict[str, Any] | None) -> Table:
        """Details tab content."""
        t = Table.grid(padding=(0, 2))
        t.add_column(style="bold cyan", no_wrap=True)
        t.add_column()
        pending = self._pending.get(info.id)
        t.add_row("Name", info.name)
        t.add_row("ID", info.short_id)
        t.add_row("Image", info.image)
        t.add_row("Engine", info.engine or "unknown")
        t.add_row("State", f"{info.emoji} {info.state}" + (f" ({info.status_text})" if info.status_text else "")
                  + (f"  ⏳ {pending}" if pending else ""))
        if info.ports:
            t.add_row("Ports", "\n".join(info.ports))
        if info.compose_project:
            t.add_row("Compose", f"{info.compose_project} / {info.labels.get('com.docker.compose.service', '?')}")
        if attrs:
            state = attrs.get("State") or {}
            started = state.get("StartedAt") or ""
            if started and not started.startswith("0001"):  # zero time = never started
                t.add_row("Started", started[:19].replace("T", " "))
            if (state.get("Health") or {}).get("Status"):
                t.add_row("Health", state["Health"]["Status"])
            if attrs.get("RestartCount"):
                t.add_row("Restarts", str(attrs["RestartCount"]))
            if state.get("ExitCode") and info.state != "running":
                t.add_row("Exit code", str(state["ExitCode"]))
            if state.get("OOMKilled"):
                t.add_row("OOM killed", "[bold red]yes[/]")
        if info.command:
            t.add_row("Command", Text(info.command))
        return t

    def _resources_view(self, report: Report, own_usage: dict[str, int]) -> Table:
        """Resources tab content: engine, limits, host GPUs and checks."""
        outer = Table.grid(padding=(1, 0))
        outer.add_column()

        def kv(title: str, rows: list[tuple[str, str]]) -> Table:
            t = Table(title=title, title_justify="left", show_header=False, box=None, padding=(0, 2))
            t.add_column(style="bold cyan", no_wrap=True)
            t.add_column()
            for k, v in rows:
                t.add_row(k, v)
            return t

        outer.add_row(kv("Engine", summarize_engine(report.engine)))
        outer.add_row(kv("Container limits", summarize_limits(report.limits)))

        snap = self.gpu_snapshot
        if snap.available:
            gt = Table(title="Host GPUs (nvidia-smi)", title_justify="left", padding=(0, 1))
            for col in ("#", "Name", "Used / Total", "Util", "This container"):
                gt.add_column(col)
            allotted = {g.uuid for g in snap.resolve(report.limits.gpus)}
            for g in snap.gpus:
                mark = "▶ " if g.uuid in allotted else "  "
                gt.add_row(
                    f"{mark}{g.index}",
                    g.name,
                    f"{g.used_mib / 1024:.1f} / {g.total_mib / 1024:.1f} GiB",
                    f"{g.util_pct}%" if g.util_pct is not None else "?",
                    f"{own_usage[g.uuid] / 1024:.1f} GiB" if g.uuid in own_usage else "",
                )
            outer.add_row(gt)
        else:
            outer.add_row(Text(f"Host GPUs: {snap.error or 'none detected'}", style="dim"))

        findings = Text()
        findings.append("Checks\n", style="bold")
        if not report.findings:
            findings.append("  ✅ no issues found\n", style="green")
        for f in report.findings:
            findings.append(f"  {f}\n", style=SEVERITY_STYLE.get(f.severity, ""))
        outer.add_row(findings)
        return outer

    def _show_compose(self, info: ContainerInfo, compose: Any, run_cmd: str | None) -> None:
        """Compose tab: the compose file if readable, else a reconstructed docker run."""
        meta = self.query_one("#compose-meta", Static)
        area = self.query_one("#compose-text", TextArea)
        if compose is not None and compose.readable:
            parts = [f"🧩 Compose project [b]{escape(compose.project)}[/] · service [b]{escape(compose.service or '?')}[/]"]
            if compose.working_dir:
                parts.append(f"   dir: {escape(compose.working_dir)}")
            for f in compose.files:
                parts.append(f"   {'✅' if f.content is not None else '❌'} {escape(f.path)}" + (f" — {escape(f.error)}" if f.error else ""))
            parts.append("   [dim]Press e to save as an editable draft.[/]")
            meta.update("\n".join(parts))
            body = "\n".join(
                (f"# ---- {f.path} ----\n" if len(compose.readable) > 1 else "") + (f.content or "")
                for f in compose.readable
            )
            set_language(area, "yaml")
            area.load_text(body)
        else:
            lines = []
            if compose is not None:
                lines.append(f"🧩 Compose project [b]{escape(compose.project)}[/] but its file is not readable here:")
                for f in compose.files:
                    lines.append(f"   ❌ {escape(f.path)} — {escape(f.error or '?')}")
            lines.append("🐳 Reconstructed [b]docker run[/] (best effort; secrets replaced by ${VAR}). Press e to save as draft.")
            meta.update("\n".join(lines))
            set_language(area, "bash")
            area.load_text(run_cmd or "")

    # -- lifecycle actions -----------------------------------------------------------

    def _target(self) -> ContainerInfo | None:
        """Highlighted container for an action, or None (with a notice) if none or busy."""
        info = self.highlighted_info()
        if info is None:
            self.notify("No container selected.", severity="warning")
            return None
        if info.id in self._pending:
            self.notify(f"{info.name} is already {self._pending[info.id]}", severity="warning")
            return None
        return info

    def action_start(self) -> None:
        """Key binding: start or unpause the highlighted container."""
        info = self._target()
        if info is None:
            return
        if info.state == "running":
            self.notify(f"{info.name} is already running.")
            return
        self._run_action(info, "start")

    def action_stop(self) -> None:
        """Key binding: stop the highlighted container after confirmation."""
        info = self._target()
        if info is None:
            return
        if info.state not in ("running", "paused", "restarting"):
            self.notify(f"{info.name} is not running.")
            return
        self.push_screen(
            ConfirmScreen(f"Stop [b]{escape(info.name)}[/]?\nThe model will have to reload when started again.", confirm_label="Stop"),
            lambda ok: ok and self._run_action(info, "stop"),
        )

    def action_restart(self) -> None:
        """Key binding: restart the highlighted container after confirmation."""
        info = self._target()
        if info is None:
            return
        self.push_screen(
            ConfirmScreen(f"Restart [b]{escape(info.name)}[/]?\nIn-flight requests will fail and weights reload.", confirm_label="Restart"),
            lambda ok: ok and self._run_action(info, "restart"),
        )

    def _run_action(self, info: ContainerInfo, verb: str) -> None:
        """Mark the container as pending and run the action in the background."""
        self._pending[info.id] = PENDING_LABEL[verb]
        item = self._item_for(info.id)
        if item is not None:
            item.set_info(self._infos.get(info.id, info), self._pending[info.id])
        self.run_worker(self._do_action(info, verb), group=f"action-{info.id}")

    @ignore_if_unmounted
    async def _do_action(self, info: ContainerInfo, verb: str) -> None:
        """Run start/stop/restart off the UI thread, then refresh the row, tabs and logs."""
        try:
            fresh = await asyncio.to_thread(getattr(self.backend, verb), info.id)
        except (DockerException, DockerUnavailable) as exc:
            self._pending.pop(info.id, None)
            self._refresh_item(info.id)
            self.notify(f"{verb.capitalize()} {info.name} failed: {exc}", severity="error", timeout=8)
            return
        self._pending.pop(info.id, None)
        self._inspect_cache.pop(info.id, None)
        current = self._infos.get(info.id, info)
        updated = dataclasses.replace(current, state=fresh.state, status_text=fresh.state)
        self._infos[info.id] = updated
        self._refresh_item(info.id)
        self.notify(f"{updated.emoji} {info.name}: {verb} → {fresh.state}")
        if self._log_container_id == info.id and fresh.state == "running" and verb in ("start", "restart"):
            self.attach_logs(info.id)  # the old follow-stream ended when the container stopped
        highlighted = self.highlighted_info()
        if highlighted is not None and highlighted.id == info.id:
            self.load_panels(updated)
        self.refresh_containers()

    def _refresh_item(self, container_id: str) -> None:
        """Redraw one sidebar row from current state."""
        item = self._item_for(container_id)
        info = self._infos.get(container_id)
        if item is not None and info is not None:
            item.set_info(info, self._pending.get(container_id))

    # -- logs ----------------------------------------------------------------------

    def action_attach_logs(self) -> None:
        """Key binding: tail the highlighted container's logs."""
        info = self.highlighted_info()
        if info is not None:
            self.attach_logs(info.id)
            self.query_one(TabbedContent).active = "tab-logs"

    def _close_log_stream(self) -> None:
        """Close the current log stream, unblocking its reader thread."""
        with self._log_lock:
            stream, self._log_stream = self._log_stream, None
        if stream is not None:
            try:
                stream.close()  # unblocks the reader thread
            except Exception:
                pass

    def attach_logs(self, container_id: str) -> None:
        """Switch the Logs tab to a container and start tailing it."""
        self._close_log_stream()
        self._log_container_id = container_id
        log = self.query_one("#log-output", Log)
        log.clear()
        info = self._infos.get(container_id)
        log.write_line(f"── logs: {info.name if info else container_id[:12]} ──")
        self._stream_logs(container_id)

    @work(thread=True, exclusive=True, group="logs")
    @ignore_if_unmounted
    def _stream_logs(self, container_id: str) -> None:
        """Background thread: read the log stream and write complete lines to the Logs tab."""
        worker = get_current_worker()
        log = self.query_one("#log-output", Log)
        try:
            stream = self.backend.open_log_stream(container_id)
        except Exception as exc:
            self.call_from_thread(log.write_line, f"Failed to stream logs: {exc}")
            return
        with self._log_lock:
            if worker.is_cancelled:
                stream.close()
                return
            self._log_stream = stream
        buffer = ""
        try:
            for chunk in stream:
                if worker.is_cancelled:
                    break
                buffer += chunk.decode("utf-8", errors="replace").replace("\r\n", "\n")
                *lines, buffer = buffer.split("\n")
                if lines:
                    self.call_from_thread(log.write_lines, lines)
        except Exception as exc:
            if not worker.is_cancelled:
                self.call_from_thread(log.write_line, f"Log stream error: {exc}")
            return
        if buffer and not worker.is_cancelled:
            self.call_from_thread(log.write_line, buffer)
        if not worker.is_cancelled:
            self.call_from_thread(log.write_line, "── log stream ended (container stopped) ──")

    # -- drafts --------------------------------------------------------------------

    def action_drafts(self, select_id: int | None = None) -> None:
        """Key binding: open the drafts screen, optionally selecting a draft."""
        if isinstance(self.screen, DraftsScreen):
            return
        self.push_screen(DraftsScreen(self.store, select_id=select_id))

    def action_save_as_draft(self) -> None:
        """Key binding: save the highlighted container as a draft."""
        info = self.highlighted_info()
        if info is None:
            self.notify("No container selected.", severity="warning")
            return
        self._prepare_draft(info)

    @work(thread=True, exclusive=True, group="save-draft")
    def _prepare_draft(self, info: ContainerInfo) -> None:
        """Build a draft from the container's compose file or a reconstructed docker run, then ask for a name."""
        try:
            attrs = self._inspect(info)
        except DockerException as exc:
            self.call_from_thread(self.notify, f"Inspect failed: {exc}", severity="error")
            return
        compose = compose_info_from_labels(info.labels)
        if compose is not None and compose.readable:
            kind = "compose"
            body = compose.readable[0].content or ""
            working_dir = compose.working_dir
            if len(compose.readable) > 1:
                self.call_from_thread(
                    self.notify, "Multiple compose files: only the first is saved in the draft.", severity="warning"
                )
        else:
            kind = "run"
            body = reconstruct_docker_run(attrs, self.backend.image_env(info.image))
            working_dir = None

        def ask() -> None:
            def created(result: dict[str, Any] | None) -> None:
                if not result:
                    return
                try:
                    draft = self.store.create(
                        self.store.unique_name(result["name"]),
                        kind,
                        body,
                        working_dir=result["working_dir"],
                        source_container=info.name,
                    )
                except ValueError as exc:
                    self.notify(str(exc), severity="error")
                    return
                self.notify(f"Saved draft '{draft.name}'.")
                self.action_drafts(select_id=draft.id)

            self.push_screen(
                DraftFormScreen(
                    name=self.store.unique_name(info.name),
                    working_dir=working_dir or "",
                    choose_template=False,
                    title=f"Save {info.name} as a {kind} draft",
                ),
                created,
            )

        self.call_from_thread(ask)

    # -- updates -------------------------------------------------------------------

    def action_check_updates(self) -> None:
        """Key binding: open the update dialog; restart the app if asked."""
        def done(restart: bool | None) -> None:
            if restart:
                self.exit("restart")

        self.push_screen(UpdateScreen(), done)

    @work(thread=True, exclusive=True, group="update-check")
    def quiet_update_check(self) -> None:
        """At most once a day, check for updates and show a notice if any."""
        try:
            last = float(self.store.get_meta("last_update_check", "0") or 0)
        except ValueError:
            last = 0
        if time.time() - last < UPDATE_CHECK_INTERVAL:
            return
        status = updater.check_for_updates()
        self.store.set_meta("last_update_check", str(time.time()))
        if status.available:
            self.call_from_thread(
                self.notify, f"{status.summary()} Press U to update.", title="Update available", timeout=10
            )
