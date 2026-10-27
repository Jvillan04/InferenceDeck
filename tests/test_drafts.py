import sqlite3

import pytest

from inferencedeck.drafts import SCHEMA_VERSION, DraftStore


def test_create_save_and_revisions(store):
    d = store.create("qwen", "run", "docker run a")
    assert len(store.revisions(d.id)) == 1
    store.save(d.id, "docker run a")  # unchanged body: no new revision
    assert len(store.revisions(d.id)) == 1
    d2 = store.save(d.id, "docker run b", working_dir="/srv/llm")
    assert d2.body == "docker run b" and d2.working_dir == "/srv/llm"
    assert [r.body for r in store.revisions(d.id)] == ["docker run b", "docker run a"]
    # working_dir is kept unless explicitly passed
    assert store.save(d.id, "docker run c").working_dir == "/srv/llm"
    assert store.save(d.id, "docker run c", working_dir=None).working_dir is None


def test_validation_and_unique_names(store):
    with pytest.raises(ValueError):
        store.create("x", "helm")
    with pytest.raises(ValueError):
        store.create("  ", "run")
    store.create("qwen", "run")
    with pytest.raises(sqlite3.IntegrityError):
        store.create("qwen", "run")
    assert store.unique_name("qwen") == "qwen-2"


def test_duplicate_and_delete_cascade(store):
    d = store.create("a", "compose", "services: {}", working_dir="/w", source_container="c1")
    copy = store.duplicate(d.id)
    assert copy.name == "a-copy" and copy.body == d.body and copy.working_dir == "/w"
    launch = store.start_launch(d.id)
    store.finish_launch(launch, 0, "ok")
    assert store.launches(d.id)[0].exit_code == 0
    store.delete(d.id)
    assert store.get(d.id) is None and store.revisions(d.id) == [] and store.launches(d.id) == []
    assert store.get(copy.id) is not None


def test_migrations_idempotent_and_meta(tmp_path):
    path = tmp_path / "x.db"
    s = DraftStore(path)
    assert s.schema_version == SCHEMA_VERSION
    s.set_meta("k", "1")
    s.set_meta("k", "2")
    s.close()
    s = DraftStore(path)
    assert s.migrate() == 0
    assert s.get_meta("k") == "2" and s.get_meta("missing", "d") == "d"
    s.close()


def test_refuses_newer_schema(tmp_path):
    path = tmp_path / "x.db"
    conn = sqlite3.connect(path)
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 5}")
    conn.close()
    with pytest.raises(RuntimeError, match="newer"):
        DraftStore(path)
