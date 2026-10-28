"""Persist Stage 4 S1 scorecards and per-factor metrics."""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.factors.expression import slugify
from src.factors.s1 import S1Result, scorecard_row
from src.factors.serialization import json_default, jsonable


@dataclass(frozen=True)
class FactorStore:
    root: Path

    def write(self, results: list[S1Result], metadata: dict[str, Any] | None = None) -> dict[str, str]:
        if not results:
            raise ValueError("cannot write an empty factor store")
        self.root.mkdir(parents=True, exist_ok=True)
        factors_dir = self.root / "factors"
        factors_dir.mkdir(exist_ok=True)

        rows = [scorecard_row(result) for result in results]
        scorecard = pd.DataFrame(rows)
        scorecard_path = self.root / "s1_scorecard.parquet"
        csv_path = self.root / "s1_scorecard.csv"
        scorecard.to_parquet(scorecard_path, index=False)
        scorecard.to_csv(csv_path, index=False)
        sqlite_path = self.root / "s1.sqlite"
        with sqlite3.connect(sqlite_path) as connection:
            scorecard.to_sql("s1_scorecard", connection, if_exists="replace", index=False)

        for result in results:
            hash_suffix = str(result.scorecard.get("expression_hash") or "error")[:8]
            factor_dir = factors_dir / f"{slugify(result.scorecard['name'])}_{hash_suffix}"
            factor_dir.mkdir(exist_ok=True)
            (factor_dir / "metrics.json").write_text(
                json.dumps(
                    jsonable(result.metrics),
                    ensure_ascii=False,
                    indent=2,
                    default=json_default,
                ),
                encoding="utf-8",
            )
            result.rank_ic.rename("rank_ic").to_csv(
                factor_dir / "rank_ic.csv", index_label="datetime"
            )

        metadata = {
            "schema_version": "stage4.s1_store.v1",
            "generated_at": datetime.now(UTC).astimezone().isoformat(timespec="seconds"),
            **(metadata or {}),
            "n_factors": len(results),
        }
        (self.root / "metadata.json").write_text(
            json.dumps(jsonable(metadata), ensure_ascii=False, indent=2, default=json_default),
            encoding="utf-8",
        )
        return {
            "parquet": str(scorecard_path),
            "csv": str(csv_path),
            "sqlite": str(sqlite_path),
        }
