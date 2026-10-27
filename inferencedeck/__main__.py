"""Command line entry point: ``inferencedeck [update|migrate|uninstall] [--version]``."""

from __future__ import annotations

import argparse
import os
import sys

from . import updater


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments and run the requested command, or start the TUI."""
    parser = argparse.ArgumentParser(prog="inferencedeck", description=__doc__)
    parser.add_argument("--version", action="store_true", help="print version and exit")
    parser.add_argument("--db", help="path to the drafts SQLite database")
    sub = parser.add_subparsers(dest="command")
    p_update = sub.add_parser("update", help="pull the latest version and reinstall")
    p_update.add_argument("--check", action="store_true", help="only report whether an update is available")
    sub.add_parser("migrate", help="apply pending drafts database migrations")
    p_uninstall = sub.add_parser("uninstall", help="remove InferenceDeck (keeps drafts unless --purge)")
    p_uninstall.add_argument("--purge", action="store_true", help="also delete the drafts database")
    p_uninstall.add_argument("-y", "--yes", action="store_true", help="don't ask for confirmation")
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

    if args.command == "uninstall":
        repo = updater.find_repo()
        script = repo / "uninstall.sh" if repo else None
        if script is None or not script.is_file():
            print("uninstall.sh not found; run it from your InferenceDeck checkout instead.", file=sys.stderr)
            return 1
        flags = (["--purge"] if args.purge else []) + (["--yes"] if args.yes else [])
        # Replace this process: the script deletes the virtualenv we're running from.
        os.execv("/bin/bash", ["bash", str(script), *flags])
        return 1  # only reached if execv is stubbed out (tests)

    if args.command == "migrate":
        from .drafts import DraftStore

        store = DraftStore(args.db)  # opening the store applies pending migrations
        print(f"{store.path}: schema v{store.schema_version} ({store.migrations_applied} migration(s) applied)")
        return 0

    from .app import InferenceDeckApp

    result = InferenceDeckApp(db_path=args.db).run()
    if result == "restart":  # requested after an in-app update: re-exec into the new code
        if os.path.isfile(sys.argv[0]) and os.access(sys.argv[0], os.X_OK):
            os.execv(sys.argv[0], sys.argv)  # the console script
        os.execv(sys.executable, [sys.executable, "-m", "inferencedeck", *sys.argv[1:]])
    return 0


if __name__ == "__main__":
    sys.exit(main())
