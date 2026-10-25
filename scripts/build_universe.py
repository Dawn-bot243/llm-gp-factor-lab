#!/usr/bin/env python
"""Build the frozen csi300 universe used by the Stage 1 baseline.

The Qlib csi300 membership file ends on 2020-09-25. Direct use would leave no
instruments after that date. This script takes the 2020-09-25 constituents and
replaces each start/end with the real coverage from `all.txt`, keeping the one
name whose coverage ends on 2020-09-25.

The output is written both into the Qlib data directory (so Qlib can load it by
name) and into `configs/universes/` (so the repo is self-contained).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
QLIB_DATA = Path("D:/qlib_data/cn_data_v3")
INSTRUMENTS = QLIB_DATA / "instruments"


def load_instruments(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["instrument", "start", "end"],
        parse_dates=["start", "end"],
    )


def main() -> None:
    csi = load_instruments(INSTRUMENTS / "csi300.txt")
    all_df = load_instruments(INSTRUMENTS / "all.txt").set_index("instrument")
    current = csi[csi["end"] == pd.Timestamp("2020-09-25")]

    rows = []
    for inst in sorted(current["instrument"].unique()):
        if inst not in all_df.index:
            continue
        real = all_df.loc[inst]
        rows.append(
            (
                inst,
                real["start"].strftime("%Y-%m-%d"),
                real["end"].strftime("%Y-%m-%d"),
            )
        )

    content = "\n".join("\t".join(row) for row in rows) + "\n"
    qlib_out = INSTRUMENTS / "csi300_frozen.txt"
    repo_out = PROJECT_ROOT / "configs" / "universes" / "csi300_frozen.txt"
    repo_out.parent.mkdir(parents=True, exist_ok=True)
    qlib_out.write_text(content, encoding="utf-8")
    repo_out.write_text(content, encoding="utf-8")
    print(f"wrote {len(rows)} instruments")
    print(f"  -> {qlib_out}")
    print(f"  -> {repo_out}")


if __name__ == "__main__":
    main()
