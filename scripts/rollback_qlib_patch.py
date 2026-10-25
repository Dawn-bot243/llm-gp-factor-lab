#!/usr/bin/env python
"""Restore Qlib bin files from a patch backup directory."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backup-dir", required=True)
    parser.add_argument("--provider-uri", default="D:/qlib_data/cn_data_v3")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    backup_dir = Path(args.backup_dir)
    provider = Path(args.provider_uri)
    files = list((backup_dir / "features").rglob("*.day.bin"))
    if not files:
        raise SystemExit(f"No bin files found under {backup_dir / 'features'}")
    for source in files:
        relative = source.relative_to(backup_dir)
        target = provider / relative
        print(f"{'would restore' if args.dry_run else 'restore'} {relative}")
        if not args.dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    print(f"{len(files)} file(s) {'checked' if args.dry_run else 'restored'} from {backup_dir}")


if __name__ == "__main__":
    main()
