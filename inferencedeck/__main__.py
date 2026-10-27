"""Command line entry point: ``inferencedeck [update|migrate] [--version]``."""

from __future__ import annotations

import argparse
import os
import sys

from . import updater


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="inferencedeck", description=__doc__)
    parser.add_argument("--version", action="store_true", help="print version and exit")
    parser.add_argument("--db", help="path to the drafts SQLite database")
    sub = parser.add_subparsers(dest="command")
    p_update = sub.add_parser("update", help="pull the latest version and reinstall")
    p_update.add_argument("--check", action="store_true", help="only report whether an update is available")
    sub.add_parser("migrate", help="apply pending drafts database migrations")
    args = parser.parse_args(argv)

    if args.version:
        print(f"inferencedeck {updater.version_string()}")
        return 0

    if args.command == "update":
        if args.check:
            status = updater.check_for_updates()
            print(status.summary())
            for c in status.commits:
                print(f"  {c}")
            return 1 if status.error else 0
        return updater.apply_update()

    if args.command == "migrate":
        from .drafts import DraftStore

        store = DraftStore(args.db)
        applied = store.migrate()
        print(f"{store.path}: schema v{store.schema_version} ({applied} migration(s) applied)")
        return 0

    from .app import InferenceDeckApp

    result = InferenceDeckApp(db_path=args.db).run()
    if result == "restart":  # requested after an in-app update: re-exec into the new code
        os.execv(sys.executable, [sys.executable, "-m", "inferencedeck", *sys.argv[1:]])
    return 0


if __name__ == "__main__":
    sys.exit(main())
