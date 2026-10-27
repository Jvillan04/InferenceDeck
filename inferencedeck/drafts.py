"""SQLite-backed store for container config drafts, their revision history and launches."""

from __future__ import annotations

import os
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

KINDS = ("run", "compose")
_UNSET = object()


def default_db_path() -> Path:
    base = os.environ.get("XDG_DATA_HOME") or os.path.join(os.path.expanduser("~"), ".local", "share")
    return Path(base) / "inferencedeck" / "inferencedeck.db"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class Draft:
    id: int
    name: str
    kind: str
    body: str
    working_dir: str | None
    notes: str
    source_container: str | None
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class Revision:
    id: int
    draft_id: int
    body: str
    saved_at: str


@dataclass(frozen=True)
class Launch:
    id: int
    draft_id: int | None
    revision_id: int | None
    started_at: str
    finished_at: str | None
    exit_code: int | None
    output_tail: str


# Each entry migrates the schema from version N to N+1. Append only; never edit old steps.
MIGRATIONS: list[str] = [
    """
    CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
    CREATE TABLE drafts (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL UNIQUE,
        kind TEXT NOT NULL CHECK (kind IN ('run', 'compose')),
        body TEXT NOT NULL DEFAULT '',
        working_dir TEXT,
        notes TEXT NOT NULL DEFAULT '',
        source_container TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );
    CREATE TABLE draft_revisions (
        id INTEGER PRIMARY KEY,
        draft_id INTEGER NOT NULL REFERENCES drafts(id) ON DELETE CASCADE,
        body TEXT NOT NULL,
        saved_at TEXT NOT NULL
    );
    CREATE INDEX idx_revisions_draft ON draft_revisions(draft_id, id);
    CREATE TABLE launches (
        id INTEGER PRIMARY KEY,
        draft_id INTEGER REFERENCES drafts(id) ON DELETE CASCADE,
        revision_id INTEGER REFERENCES draft_revisions(id) ON DELETE SET NULL,
        started_at TEXT NOT NULL,
        finished_at TEXT,
        exit_code INTEGER,
        output_tail TEXT NOT NULL DEFAULT ''
    );
    """,
]

SCHEMA_VERSION = len(MIGRATIONS)


class DraftStore:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else default_db_path()
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self.migrations_applied = self.migrate()

    # -- schema --------------------------------------------------------------------

    @property
    def schema_version(self) -> int:
        return self._conn.execute("PRAGMA user_version").fetchone()[0]

    def migrate(self) -> int:
        """Bring the schema up to date. Returns the number of migrations applied."""
        with self._lock:
            current = self.schema_version
            if current > SCHEMA_VERSION:
                raise RuntimeError(
                    f"Database schema v{current} is newer than this InferenceDeck (v{SCHEMA_VERSION}); update the app."
                )
            for version in range(current, SCHEMA_VERSION):
                with self._conn:
                    self._conn.executescript(f"BEGIN;\n{MIGRATIONS[version]}\nPRAGMA user_version = {version + 1};\nCOMMIT;")
            return SCHEMA_VERSION - current

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # -- meta ----------------------------------------------------------------------

    def get_meta(self, key: str, default: str | None = None) -> str | None:
        with self._lock:
            row = self._conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else default

    def set_meta(self, key: str, value: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )

    # -- drafts --------------------------------------------------------------------

    def list_drafts(self) -> list[Draft]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM drafts ORDER BY updated_at DESC, name").fetchall()
        return [Draft(**dict(r)) for r in rows]

    def get(self, draft_id: int) -> Draft | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM drafts WHERE id = ?", (draft_id,)).fetchone()
        return Draft(**dict(row)) if row else None

    def get_by_name(self, name: str) -> Draft | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM drafts WHERE name = ?", (name,)).fetchone()
        return Draft(**dict(row)) if row else None

    def unique_name(self, base: str) -> str:
        base = base.strip() or "draft"
        name, n = base, 2
        while self.get_by_name(name) is not None:
            name = f"{base}-{n}"
            n += 1
        return name

    def create(
        self,
        name: str,
        kind: str,
        body: str = "",
        *,
        working_dir: str | None = None,
        notes: str = "",
        source_container: str | None = None,
    ) -> Draft:
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        name = name.strip()
        if not name:
            raise ValueError("name must not be empty")
        now = _now()
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO drafts(name, kind, body, working_dir, notes, source_container, created_at, updated_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (name, kind, body, working_dir, notes, source_container, now, now),
            )
            draft_id = cur.lastrowid
            self._conn.execute(
                "INSERT INTO draft_revisions(draft_id, body, saved_at) VALUES (?, ?, ?)", (draft_id, body, now)
            )
        return self.get(draft_id)  # type: ignore[return-value]

    def save(
        self,
        draft_id: int,
        body: str,
        *,
        working_dir: str | None | object = _UNSET,
        notes: str | None = None,
    ) -> Draft:
        """Save new content; a revision is recorded only when the body actually changed."""
        current = self.get(draft_id)
        if current is None:
            raise KeyError(draft_id)
        now = _now()
        with self._lock, self._conn:
            if body != current.body:
                self._conn.execute(
                    "INSERT INTO draft_revisions(draft_id, body, saved_at) VALUES (?, ?, ?)", (draft_id, body, now)
                )
            self._conn.execute(
                "UPDATE drafts SET body = ?, working_dir = ?, notes = ?, updated_at = ? WHERE id = ?",
                (
                    body,
                    current.working_dir if working_dir is _UNSET else working_dir,
                    current.notes if notes is None else notes,
                    now,
                    draft_id,
                ),
            )
        return self.get(draft_id)  # type: ignore[return-value]

    def rename(self, draft_id: int, name: str) -> Draft:
        with self._lock, self._conn:
            self._conn.execute("UPDATE drafts SET name = ?, updated_at = ? WHERE id = ?", (name.strip(), _now(), draft_id))
        return self.get(draft_id)  # type: ignore[return-value]

    def duplicate(self, draft_id: int, name: str | None = None) -> Draft:
        src = self.get(draft_id)
        if src is None:
            raise KeyError(draft_id)
        return self.create(
            name or self.unique_name(f"{src.name}-copy"),
            src.kind,
            src.body,
            working_dir=src.working_dir,
            notes=src.notes,
            source_container=src.source_container,
        )

    def delete(self, draft_id: int) -> None:
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM drafts WHERE id = ?", (draft_id,))

    # -- revisions -----------------------------------------------------------------

    def revisions(self, draft_id: int) -> list[Revision]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM draft_revisions WHERE draft_id = ? ORDER BY id DESC", (draft_id,)
            ).fetchall()
        return [Revision(**dict(r)) for r in rows]

    def latest_revision(self, draft_id: int) -> Revision | None:
        revs = self.revisions(draft_id)
        return revs[0] if revs else None

    # -- launches ------------------------------------------------------------------

    def start_launch(self, draft_id: int) -> int:
        rev = self.latest_revision(draft_id)
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO launches(draft_id, revision_id, started_at) VALUES (?, ?, ?)",
                (draft_id, rev.id if rev else None, _now()),
            )
        return cur.lastrowid  # type: ignore[return-value]

    def finish_launch(self, launch_id: int, exit_code: int, output_tail: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE launches SET finished_at = ?, exit_code = ?, output_tail = ? WHERE id = ?",
                (_now(), exit_code, output_tail[-8000:], launch_id),
            )

    def launches(self, draft_id: int) -> list[Launch]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM launches WHERE draft_id = ? ORDER BY id DESC", (draft_id,)
            ).fetchall()
        return [Launch(**dict(r)) for r in rows]
