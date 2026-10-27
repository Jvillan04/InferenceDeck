"""Modal dialogs and the drafts editor screen."""

from __future__ import annotations

import difflib
from typing import TYPE_CHECKING, Any

from rich.markup import escape
from rich.text import Text
from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen, Screen
from textual.timer import Timer
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Label,
    ListItem,
    ListView,
    Log,
    Select,
    Static,
    TextArea,
)

from . import updater
from .drafts import Draft, DraftStore, Revision
from .launcher import LaunchError, plan_launch, run_plan
from .lint import Report, count_by_severity, lint_draft
from .templates import TEMPLATES

if TYPE_CHECKING:
    from .app import InferenceDeckApp

SEVERITY_STYLE = {"error": "bold red", "warn": "yellow", "info": "cyan", "ok": "green"}
KIND_ICON = {"run": "🐳", "compose": "🧩"}


def set_language(area: TextArea, language: str | None) -> None:
    """Enable syntax highlighting when tree-sitter support is installed."""
    try:
        area.language = language if language in area.available_languages else None
    except Exception:  # LanguageDoesNotExist and friends: plain text is fine
        area.language = None


def language_for(kind: str) -> str:
    return "yaml" if kind == "compose" else "bash"


def render_reports(reports: list[Report]) -> Text:
    text = Text()
    for i, report in enumerate(reports):
        if i:
            text.append("\n")
        text.append(f"{report.scope}\n", style="bold")
        if not report.findings:
            text.append("  ✅ no issues found\n", style="green")
        for f in report.findings:
            text.append(f"  {f}\n", style=SEVERITY_STYLE.get(f.severity, ""))
    return text


# --------------------------------------------------------------------------------
# Small modals
# --------------------------------------------------------------------------------


class ConfirmScreen(ModalScreen[bool]):
    BINDINGS = [
        Binding("y", "answer(True)", "Yes"),
        Binding("n,escape", "answer(False)", "No"),
    ]

    def __init__(self, message: str, *, confirm_label: str = "Yes", danger: bool = True) -> None:
        super().__init__()
        self.message = message
        self.confirm_label = confirm_label
        self.danger = danger

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog"):
            yield Static(self.message, classes="dialog-body")
            with Horizontal(classes="dialog-buttons"):
                yield Button(f"{self.confirm_label} (y)", variant="error" if self.danger else "primary", id="yes")
                yield Button("Cancel (n)", id="no")

    def on_mount(self) -> None:
        self.query_one("#no", Button).focus()

    @on(Button.Pressed)
    def _pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "yes")

    def action_answer(self, value: bool) -> None:
        self.dismiss(value)


class PromptScreen(ModalScreen["str | None"]):
    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    def __init__(self, title: str, value: str = "", placeholder: str = "") -> None:
        super().__init__()
        self.title_text = title
        self.value = value
        self.placeholder = placeholder

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog"):
            yield Label(self.title_text, classes="dialog-title")
            yield Input(self.value, placeholder=self.placeholder, id="prompt-input")
            with Horizontal(classes="dialog-buttons"):
                yield Button("OK", variant="primary", id="ok")
                yield Button("Cancel", id="cancel")

    @on(Input.Submitted)
    @on(Button.Pressed, "#ok")
    def _submit(self) -> None:
        self.dismiss(self.query_one("#prompt-input", Input).value)

    @on(Button.Pressed, "#cancel")
    def action_cancel(self) -> None:
        self.dismiss(None)


class DraftFormScreen(ModalScreen["dict[str, Any] | None"]):
    """Name + template (+ working dir) for a new draft."""

    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    def __init__(self, *, name: str = "", working_dir: str = "", choose_template: bool = True, title: str = "New draft") -> None:
        super().__init__()
        self.default_name = name
        self.default_wd = working_dir
        self.choose_template = choose_template
        self.title_text = title

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog"):
            yield Label(self.title_text, classes="dialog-title")
            yield Label("Name")
            yield Input(self.default_name, placeholder="e.g. qwen-7b-tp2", id="name")
            if self.choose_template:
                yield Label("Template")
                yield Select(
                    [(label, i) for i, (label, _, _) in enumerate(TEMPLATES)], value=0, allow_blank=False, id="template"
                )
            yield Label("Working directory (for relative paths / compose project)")
            yield Input(self.default_wd, placeholder="optional, e.g. ~/models/qwen", id="workdir")
            with Horizontal(classes="dialog-buttons"):
                yield Button("Create", variant="primary", id="create")
                yield Button("Cancel", id="cancel")

    @on(Input.Submitted)
    @on(Button.Pressed, "#create")
    def _submit(self) -> None:
        name = self.query_one("#name", Input).value.strip()
        if not name:
            self.notify("Name is required.", severity="warning")
            return
        result: dict[str, Any] = {"name": name, "working_dir": self.query_one("#workdir", Input).value.strip() or None}
        if self.choose_template:
            result["template"] = self.query_one("#template", Select).value
        self.dismiss(result)

    @on(Button.Pressed, "#cancel")
    def action_cancel(self) -> None:
        self.dismiss(None)


class RevisionsScreen(ModalScreen["str | None"]):
    """Browse a draft's revision history; Enter restores the highlighted revision into the editor."""

    BINDINGS = [
        Binding("escape", "cancel", "Close"),
        Binding("d", "toggle_diff", "Diff/full"),
        Binding("enter", "restore", "Restore", show=True),
    ]

    def __init__(self, revisions: list[Revision], current_text: str) -> None:
        super().__init__()
        self.revisions = revisions
        self.current_text = current_text
        self.show_diff = True

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog wide"):
            yield Label("Revision history — Enter restores into the editor (not saved until Ctrl+S)", classes="dialog-title")
            with Horizontal(id="rev-body"):
                yield ListView(
                    *[ListItem(Label(f"#{len(self.revisions) - i}  {r.saved_at[:19].replace('T', ' ')} UTC")) for i, r in enumerate(self.revisions)],
                    id="rev-list",
                )
                yield TextArea("", read_only=True, id="rev-preview", show_line_numbers=False)
            yield Footer()

    def on_mount(self) -> None:
        self._preview(0)

    def _preview(self, index: int | None) -> None:
        if index is None or not self.revisions:
            return
        rev = self.revisions[index]
        area = self.query_one("#rev-preview", TextArea)
        if self.show_diff:
            diff = difflib.unified_diff(
                rev.body.splitlines(), self.current_text.splitlines(), "revision", "editor", lineterm=""
            )
            area.load_text("\n".join(diff) or "(identical to editor)")
            set_language(area, None)
        else:
            area.load_text(rev.body)

    @on(ListView.Highlighted, "#rev-list")
    def _highlighted(self, event: ListView.Highlighted) -> None:
        self._preview(event.list_view.index)

    @on(ListView.Selected, "#rev-list")
    def action_restore(self) -> None:
        index = self.query_one("#rev-list", ListView).index
        if index is not None and self.revisions:
            self.dismiss(self.revisions[index].body)

    def action_toggle_diff(self) -> None:
        self.show_diff = not self.show_diff
        self._preview(self.query_one("#rev-list", ListView).index)

    def action_cancel(self) -> None:
        self.dismiss(None)


class UpdateScreen(ModalScreen[bool]):
    """Check for updates, apply them, and offer a restart. Dismisses True to request a restart."""

    BINDINGS = [Binding("escape", "close", "Close")]

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog wide"):
            yield Label("InferenceDeck updates", classes="dialog-title")
            yield Static(f"Installed: {updater.version_string()}\nChecking for updates…", id="update-status")
            yield Log(id="update-log", max_lines=2000)
            with Horizontal(classes="dialog-buttons"):
                yield Button("Update", variant="success", id="update", disabled=True)
                yield Button("Restart now", variant="primary", id="restart", classes="hidden")
                yield Button("Close", id="close")

    def on_mount(self) -> None:
        self.check()

    @work(thread=True, exclusive=True, group="update")
    def check(self) -> None:
        status = updater.check_for_updates()
        lines = [f"Installed: {updater.version_string()}", status.summary()]
        if status.dirty:
            lines.append("Local changes in the install checkout: update disabled.")
        self.app.call_from_thread(self.query_one("#update-status", Static).update, "\n".join(lines))
        log = self.query_one("#update-log", Log)
        self.app.call_from_thread(log.write_lines, status.commits)
        if status.available and not status.dirty:
            self.app.call_from_thread(setattr, self.query_one("#update", Button), "disabled", False)

    @on(Button.Pressed, "#update")
    def _update(self) -> None:
        self.query_one("#update", Button).disabled = True
        self.apply()

    @work(thread=True, exclusive=True, group="update")
    def apply(self) -> None:
        log = self.query_one("#update-log", Log)
        self.app.call_from_thread(log.clear)
        code = updater.apply_update(on_line=lambda line: self.app.call_from_thread(log.write_line, line))
        if code == 0:
            self.app.call_from_thread(self.query_one("#restart", Button).remove_class, "hidden")

    @on(Button.Pressed, "#restart")
    def _restart(self) -> None:
        self.dismiss(True)

    @on(Button.Pressed, "#close")
    def action_close(self) -> None:
        self.dismiss(False)


# --------------------------------------------------------------------------------
# Drafts
# --------------------------------------------------------------------------------


class DraftItem(ListItem):
    def __init__(self, draft: Draft, dirty: bool) -> None:
        self.draft_id = draft.id
        self.label = Label(self.render_label(draft, dirty))
        super().__init__(self.label)

    @staticmethod
    def render_label(draft: Draft, dirty: bool) -> str:
        mark = "[bold yellow]●[/] " if dirty else ""
        return f"{mark}{KIND_ICON.get(draft.kind, '')} {escape(draft.name)}\n   [dim]{draft.updated_at.replace('T', ' ')[:16]}[/]"

    def refresh_label(self, draft: Draft, dirty: bool) -> None:
        self.label.update(self.render_label(draft, dirty))


class DraftsScreen(Screen):
    BINDINGS = [
        Binding("escape", "back", "Back", priority=True),  # beats TextArea's escape-to-unfocus
        Binding("ctrl+s", "save", "Save"),
        Binding("ctrl+n", "new", "New"),
        Binding("ctrl+l", "launch", "Launch"),
        Binding("ctrl+t", "duplicate", "Duplicate"),
        Binding("ctrl+r", "history", "History"),
        Binding("ctrl+g", "working_dir", "Work dir"),
        Binding("f8", "abort_launch", "Abort launch", show=False),
        Binding("delete", "delete", "Delete", show=False),
    ]

    def __init__(self, store: DraftStore, select_id: int | None = None) -> None:
        super().__init__()
        self.store = store
        self._select_id = select_id
        self._drafts: dict[int, Draft] = {}
        self._buffers: dict[int, str] = {}  # unsaved editor contents per draft
        self._current_id: int | None = None
        self._loading = False
        self._lint_timer: Timer | None = None
        self._launch_proc: Any = None

    @property
    def deck(self) -> "InferenceDeckApp":
        return self.app  # type: ignore[return-value]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal():
            with Vertical(id="drafts-sidebar"):
                yield Label("📝 Drafts", classes="sidebar-title")
                yield ListView(id="draft-list")
                yield Static("Ctrl+N new · Del delete", classes="hint")
            with Vertical(id="drafts-main"):
                yield Label("No draft selected — Ctrl+N to create one", id="draft-title")
                yield TextArea.code_editor("", id="editor", soft_wrap=True, tab_behavior="indent")
                with VerticalScroll(id="lint-panel"):
                    yield Static("", id="lint")
                yield Log(id="launch-log", max_lines=2000)
        yield Footer()

    async def on_mount(self) -> None:
        self.sub_title = "Drafts"
        self.query_one("#editor", TextArea).disabled = True
        await self.reload_list(select_id=self._select_id)
        self.query_one("#draft-list", ListView).focus()

    # -- list ----------------------------------------------------------------------

    async def reload_list(self, select_id: int | None = None) -> None:
        drafts = self.store.list_drafts()
        self._drafts = {d.id: d for d in drafts}
        lv = self.query_one("#draft-list", ListView)
        if select_id is None:
            select_id = self._current_id
        await lv.clear()
        await lv.extend(DraftItem(d, d.id in self._buffers) for d in drafts)
        if not drafts:
            self._show(None)
            return
        ids = [d.id for d in drafts]
        lv.index = ids.index(select_id) if select_id in ids else 0
        self._show(ids[lv.index])

    def _refresh_item(self, draft_id: int) -> None:
        draft = self._drafts.get(draft_id)
        for item in self.query(DraftItem):
            if item.draft_id == draft_id and draft is not None:
                item.refresh_label(draft, draft_id in self._buffers)

    @on(ListView.Highlighted, "#draft-list")
    def _on_highlight(self, event: ListView.Highlighted) -> None:
        if isinstance(event.item, DraftItem) and event.item.draft_id != self._current_id:
            self._show(event.item.draft_id)

    @on(ListView.Selected, "#draft-list")
    def _on_select(self) -> None:
        self.query_one("#editor", TextArea).focus()

    # -- editor --------------------------------------------------------------------

    def _show(self, draft_id: int | None) -> None:
        editor = self.query_one("#editor", TextArea)
        self._current_id = draft_id
        draft = self._drafts.get(draft_id) if draft_id is not None else None
        self._loading = True
        if draft is None:
            editor.load_text("")
            editor.disabled = True
            self.query_one("#draft-title", Label).update("No draft selected — Ctrl+N to create one")
            self.query_one("#lint", Static).update("")
        else:
            editor.disabled = False
            set_language(editor, language_for(draft.kind))
            editor.load_text(self._buffers.get(draft.id, draft.body))
            self._update_title()
            self._lint_now()
        self.call_after_refresh(setattr, self, "_loading", False)

    def _update_title(self) -> None:
        draft = self._drafts.get(self._current_id) if self._current_id is not None else None
        if draft is None:
            return
        dirty = " [bold yellow]● unsaved[/]" if draft.id in self._buffers else ""
        wd = f" · [dim]{escape(draft.working_dir)}[/]" if draft.working_dir else ""
        src = f" · from [dim]{escape(draft.source_container)}[/]" if draft.source_container else ""
        self.query_one("#draft-title", Label).update(
            f"{KIND_ICON.get(draft.kind, '')} [b]{escape(draft.name)}[/] ({draft.kind}){wd}{src}{dirty}"
        )

    @on(TextArea.Changed, "#editor")
    def _on_edit(self) -> None:
        if self._loading or self._current_id is None:
            return
        draft = self._drafts[self._current_id]
        text = self.query_one("#editor", TextArea).text
        was_dirty = draft.id in self._buffers
        if text == draft.body:
            self._buffers.pop(draft.id, None)
        else:
            self._buffers[draft.id] = text
        if was_dirty != (draft.id in self._buffers):
            self._refresh_item(draft.id)
            self._update_title()
        if self._lint_timer is not None:
            self._lint_timer.stop()
        self._lint_timer = self.set_timer(0.4, self._lint_now)

    def _lint_reports(self) -> list[Report]:
        draft = self._drafts[self._current_id]  # type: ignore[index]
        text = self.query_one("#editor", TextArea).text
        return lint_draft(
            draft.kind,
            text,
            self.deck.gpu_snapshot,
            peers=self.deck.peers,
            existing_names=self.deck.container_names,
        )

    def _lint_now(self) -> None:
        if self._current_id is None or self._current_id not in self._drafts:
            return
        self.query_one("#lint", Static).update(render_reports(self._lint_reports()))

    # -- actions -------------------------------------------------------------------

    def action_save(self) -> None:
        if self._current_id is None:
            return
        text = self.query_one("#editor", TextArea).text
        draft = self.store.save(self._current_id, text)
        self._drafts[draft.id] = draft
        self._buffers.pop(draft.id, None)
        self._refresh_item(draft.id)
        self._update_title()
        self.notify(f"Saved '{draft.name}' (revision {len(self.store.revisions(draft.id))}).")

    def action_new(self) -> None:
        def created(result: dict[str, Any] | None) -> None:
            if not result:
                return
            _, kind, body = TEMPLATES[result["template"]]
            try:
                draft = self.store.create(
                    self.store.unique_name(result["name"]), kind, body, working_dir=result["working_dir"]
                )
            except ValueError as exc:
                self.notify(str(exc), severity="error")
                return
            self.run_worker(self.reload_list(select_id=draft.id))
            self.query_one("#editor", TextArea).focus()

        self.app.push_screen(DraftFormScreen(), created)

    async def action_duplicate(self) -> None:
        if self._current_id is None:
            return
        copy = self.store.duplicate(self._current_id)
        if self._current_id in self._buffers:  # carry unsaved edits into the copy
            copy = self.store.save(copy.id, self._buffers[self._current_id])
        await self.reload_list(select_id=copy.id)
        self.notify(f"Duplicated as '{copy.name}'.")

    def action_delete(self) -> None:
        if self._current_id is None:
            return
        draft = self._drafts[self._current_id]

        def confirmed(ok: bool | None) -> None:
            if ok:
                self.store.delete(draft.id)
                self._buffers.pop(draft.id, None)
                self._current_id = None
                self.run_worker(self.reload_list())
                self.notify(f"Deleted '{draft.name}'.")

        self.app.push_screen(
            ConfirmScreen(f"Delete draft [b]{escape(draft.name)}[/] and its whole revision history?", confirm_label="Delete"),
            confirmed,
        )

    def action_history(self) -> None:
        if self._current_id is None:
            return
        editor = self.query_one("#editor", TextArea)

        def restore(body: str | None) -> None:
            if body is not None:
                editor.load_text(body)  # fires Changed -> marks unsaved
                editor.focus()

        self.app.push_screen(RevisionsScreen(self.store.revisions(self._current_id), editor.text), restore)

    def action_working_dir(self) -> None:
        if self._current_id is None:
            return
        draft = self._drafts[self._current_id]

        def chosen(value: str | None) -> None:
            if value is None:
                return
            updated = self.store.save(draft.id, draft.body, working_dir=value.strip() or None)
            self._drafts[draft.id] = updated
            self._update_title()

        self.app.push_screen(
            PromptScreen("Working directory (compose project dir / cwd for docker run)", draft.working_dir or ""), chosen
        )

    def action_launch(self) -> None:
        if self._current_id is None:
            return
        if self._current_id in self._buffers:
            self.action_save()
        draft = self._drafts[self._current_id]
        counts = count_by_severity(self._lint_reports())
        summary = (
            f"[red]{counts['error']} error(s)[/], [yellow]{counts['warn']} warning(s)[/] found."
            if counts["error"] or counts["warn"]
            else "[green]No issues found.[/]"
        )
        what = "docker run (via sh -c)" if draft.kind == "run" else "docker compose up -d"
        self.app.push_screen(
            ConfirmScreen(
                f"Launch [b]{escape(draft.name)}[/] with {what}?\n\n{summary}",
                confirm_label="Launch",
                danger=bool(counts["error"]),
            ),
            lambda ok: ok and self._launch(draft),
        )

    @work(thread=True, exclusive=True, group="launch")
    def _launch(self, draft: Draft) -> None:
        log = self.query_one("#launch-log", Log)
        write = lambda line: self.app.call_from_thread(log.write_line, line)  # noqa: E731
        self.app.call_from_thread(log.clear)
        try:
            plan = plan_launch(draft)
        except LaunchError as exc:
            write(f"✖ {exc}")
            self.app.call_from_thread(self.notify, str(exc), severity="error")
            return
        for w in plan.warnings:
            write(f"⚠ {w}")
        write(f"▶ {plan.display}" + (f"  (cwd {plan.cwd})" if plan.cwd else ""))
        launch_id = self.store.start_launch(draft.id)
        output: list[str] = []

        def on_line(line: str) -> None:
            output.append(line)
            write(line)

        code = run_plan(plan, on_line, on_start=lambda proc: setattr(self, "_launch_proc", proc))
        self._launch_proc = None
        self.store.finish_launch(launch_id, code, "\n".join(output))
        if code == 0:
            write("✔ launched")
            self.app.call_from_thread(self.notify, f"Launched '{draft.name}'.")
        else:
            write(f"✖ exited with code {code}")
            self.app.call_from_thread(self.notify, f"Launch of '{draft.name}' failed ({code}).", severity="error")
        self.app.call_from_thread(self.deck.refresh_containers)

    def action_abort_launch(self) -> None:
        proc = self._launch_proc
        if proc is not None and proc.poll() is None:
            proc.terminate()
            self.notify("Launch aborted (the container may still have been created).", severity="warning")

    def action_back(self) -> None:
        if not self._buffers:
            self.app.pop_screen()
            return
        names = ", ".join(self._drafts[i].name for i in self._buffers if i in self._drafts)

        def confirmed(ok: bool | None) -> None:
            if ok:
                self.app.pop_screen()

        self.app.push_screen(
            ConfirmScreen(f"Discard unsaved changes in: {escape(names)}?", confirm_label="Discard"), confirmed
        )
