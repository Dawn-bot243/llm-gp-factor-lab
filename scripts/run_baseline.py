#!/usr/bin/env python
"""Stage 1 baseline: Alpha158 + LightGBM + TopkDropout.

This script deliberately keeps the baseline simple and reproducible:
- one Qlib data snapshot
- one time-based train/valid/test split
- one LightGBM model
- one TopkDropout portfolio
- factor-level metrics (RankIC/ICIR/quantile/turnover) plus portfolio metrics

Run a quick smoke version first:
    .venv/Scripts/python.exe scripts/run_baseline.py --quick
Then the full version:
    .venv/Scripts/python.exe scripts/run_baseline.py
"""

from __future__ import annotations

import os

# Qlib's LGBModel logs to MLflow. Newer MLflow versions reject the default
# filesystem backend unless this opt-in is set. Must be set before importing qlib.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qlib.config import C


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "baseline.yaml"))
    parser.add_argument("--quick", action="store_true", help="Small csi100/2-year smoke run")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--instruments", default=None)
    parser.add_argument("--start", default=None)
    parser.add_argument("--train-end", default=None)
    parser.add_argument("--valid-end", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument("--n-drop", type=int, default=None)
    parser.add_argument("--num-threads", type=int, default=None)
    parser.add_argument(
        "--mask-file",
        default=None,
        help="JSON from scripts/build_quality_mask.py; excludes unusable dates/instruments",
    )
    parser.add_argument(
        "--snapshot-file",
        default=str(PROJECT_ROOT / "data" / "manifests" / "qlib_snapshot.json"),
        help="Qlib data snapshot manifest; its id is recorded in metrics.json",
    )
    parser.add_argument(
        "--pit-membership",
        default=None,
        help="Parquet with daily PIT members (instrument, datetime); overrides --mask-file",
    )
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> dict[str, Any]:
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.quick:
        cfg["data"].update(
            {
                "instruments": "csi300",
                "start_time": "2019-01-01",
                "train_end": "2019-12-31",
                "valid_end": "2020-06-30",
                "end_time": "2020-09-23",
                "fit_start_time": "2019-01-01",
                "fit_end_time": "2019-12-31",
            }
        )
        cfg["model"]["num_boost_round"] = 100
        cfg["backtest"]["topk"] = 30
        cfg["backtest"]["n_drop"] = 3

    for key, value in {
        "instruments": args.instruments,
        "start_time": args.start,
        "train_end": args.train_end,
        "valid_end": args.valid_end,
        "end_time": args.end,
    }.items():
        if value is not None:
            cfg["data"][key] = value

    if args.topk is not None:
        cfg["backtest"]["topk"] = args.topk
    if args.n_drop is not None:
        cfg["backtest"]["n_drop"] = args.n_drop
    if args.num_threads is not None:
        cfg["model"]["num_threads"] = args.num_threads

    return cfg


def _normalize_series(s: pd.Series, name: str) -> pd.Series:
    s = s.copy()
    s.name = name
    names = list(s.index.names)
    if names == ["datetime", "instrument"]:
        s = s.reorder_levels(["instrument", "datetime"])
    elif names != ["instrument", "datetime"]:
        raise ValueError(f"Unexpected index names: {names}")
    return s.sort_index()


def compute_ic(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    rank_ics: dict[pd.Timestamp, float] = {}
    pearson_ics: dict[pd.Timestamp, float] = {}
    for dt, g in df.groupby(level="datetime", sort=True):
        if len(g) < 20:
            continue
        rank_ics[dt] = float(g["pred"].corr(g["label"], method="spearman"))
        pearson_ics[dt] = float(g["pred"].corr(g["label"], method="pearson"))
    return (
        pd.Series(rank_ics, name="rank_ic").sort_index(),
        pd.Series(pearson_ics, name="ic").sort_index(),
    )


def ic_stats(ic: pd.Series) -> dict[str, Any]:
    ic = ic.dropna()
    if ic.empty:
        return {}
    std = float(ic.std(ddof=1))
    mean = float(ic.mean())
    icir = mean / std if std > 0 else float("nan")
    t_stat = mean / std * np.sqrt(len(ic)) if std > 0 else float("nan")
    newey_west_t = None
    try:
        import statsmodels.api as sm

        x = np.ones((len(ic), 1))
        fit = sm.OLS(ic.values, x).fit(
            cov_type="HAC", cov_kwds={"maxlags": min(10, max(1, len(ic) - 1))}
        )
        newey_west_t = float(fit.tvalues[0])
    except Exception as exc:  # pragma: no cover - diagnostics only
        newey_west_t = f"error: {exc!r}"
    return {
        "n_dates": len(ic),
        "mean": mean,
        "std": std,
        "icir": float(icir),
        "icir_annualized": float(icir * np.sqrt(252)) if np.isfinite(icir) else None,
        "t_stat": float(t_stat),
        "newey_west_t": newey_west_t,
        "positive_ratio": float((ic > 0).mean()),
    }


def quantile_stats(df: pd.DataFrame, q: int = 5) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    rows = []
    for _, g in df.groupby(level="datetime", sort=True):
        if len(g) < q * 2:
            continue
        ranks = g["pred"].rank(method="first")
        buckets = pd.qcut(ranks, q, labels=False)
        rows.append(g.assign(bucket=buckets).groupby("bucket")["label"].mean())
    if not rows:
        return pd.Series(dtype=float), pd.Series(dtype=float), pd.DataFrame()
    qdf = pd.DataFrame(rows)
    qmean = qdf.mean()
    long_short = qdf[q - 1] - qdf[0]
    return qmean, long_short, qdf


def top_quantile_turnover(df: pd.DataFrame, q: int = 5) -> float:
    sets: list[set[str]] = []
    for _, g in df.groupby(level="datetime", sort=True):
        if len(g) < q * 2:
            continue
        k = max(1, len(g) // q)
        top = g.nlargest(k, "pred").index.get_level_values("instrument")
        sets.append(set(top))
    turnovers = []
    for prev, cur in itertools.pairwise(sets):
        if prev:
            turnovers.append(1.0 - len(prev & cur) / len(prev))
    return float(np.mean(turnovers)) if turnovers else float("nan")


def backtest_metrics(report_normal: pd.DataFrame) -> dict[str, Any]:
    from qlib.contrib.evaluate import risk_analysis

    series_map = {
        "portfolio": report_normal["return"],
        "excess_without_cost": report_normal["return"] - report_normal["bench"],
        "excess_with_cost": report_normal["return"] - report_normal["bench"] - report_normal["cost"],
    }
    out: dict[str, Any] = {}
    for name, series in series_map.items():
        try:
            out[name] = risk_analysis(series.rename(name), freq="day").to_dict()
        except Exception as exc:  # pragma: no cover - diagnostics only
            out[name] = {"error": repr(exc)}
    return out


def save_plots(rank_ic: pd.Series, qmean: pd.Series, report_normal: pd.DataFrame, out_dir: Path) -> None:
    if not rank_ic.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        rank_ic.cumsum().plot(ax=ax, title="Cumulative RankIC (test)")
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.set_ylabel("cumulative RankIC")
        fig.tight_layout()
        fig.savefig(out_dir / "cumulative_rank_ic.png", dpi=150)
        plt.close(fig)

    if not qmean.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        qmean.plot(kind="bar", ax=ax, title="Mean next-day return by factor quantile")
        ax.set_xlabel("quantile (0 = lowest factor)")
        ax.set_ylabel("mean forward return")
        fig.tight_layout()
        fig.savefig(out_dir / "quantile_returns.png", dpi=150)
        plt.close(fig)

    if not report_normal.empty:
        cum = (1.0 + report_normal["return"]).cumprod()
        cum_bench = (1.0 + report_normal["bench"]).cumprod()
        fig, ax = plt.subplots(figsize=(10, 4))
        cum.plot(ax=ax, label="strategy")
        cum_bench.plot(ax=ax, label="benchmark")
        ax.set_title("Cumulative return (test, after cost)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "cumulative_return.png", dpi=150)
        plt.close(fig)

        drawdown = cum / cum.cummax() - 1.0
        fig, ax = plt.subplots(figsize=(10, 3))
        drawdown.plot(ax=ax, title="Strategy drawdown")
        ax.set_ylabel("drawdown")
        fig.tight_layout()
        fig.savefig(out_dir / "drawdown.png", dpi=150)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    cfg = load_config(args)
    out_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "artifacts" / (
        "baseline_quick" if args.quick else "baseline"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    C.joblib_backend = cfg["runtime"]["joblib_backend"]
    C.kernels = int(cfg["runtime"]["kernels"])

    import qlib
    from qlib.backtest import backtest
    from qlib.contrib.data.handler import Alpha158
    from qlib.contrib.model.gbdt import LGBModel
    from qlib.contrib.strategy import TopkDropoutStrategy
    from qlib.data import D
    from qlib.data.dataset import DatasetH

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    bt_cfg = cfg["backtest"]

    snapshot: dict[str, Any] | None = None
    snapshot_path = Path(args.snapshot_file)
    if snapshot_path.exists():
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))

    mask: dict[str, Any] | None = None
    exclude_dates = pd.DatetimeIndex([])
    exclude_instruments: set[str] = set()
    if args.mask_file:
        mask = json.loads(Path(args.mask_file).read_text(encoding="utf-8"))
        exclude_dates = pd.DatetimeIndex(pd.to_datetime(mask.get("excluded_dates", [])))
        exclude_instruments = {str(code).upper() for code in mask.get("excluded_instruments", [])}
        print(f"=== mask: {len(exclude_dates)} dates, {len(exclude_instruments)} instruments ===")

    def _next_day(value: str) -> str:
        return (pd.Timestamp(value) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    # Qlib slices are inclusive; start valid/test the day after the previous
    # segment ends to avoid a one-day overlap.
    segments = {
        "train": (data_cfg["start_time"], data_cfg["train_end"]),
        "valid": (_next_day(data_cfg["train_end"]), data_cfg["valid_end"]),
        "test": (_next_day(data_cfg["valid_end"]), data_cfg["end_time"]),
    }

    print("=== config ===")
    print(json.dumps({"data": data_cfg, "segments": segments, "model": model_cfg, "backtest": bt_cfg}, indent=2))

    qlib.init(provider_uri=data_cfg["provider_uri"], region=data_cfg["region"])
    # qlib.init() calls C.set(...), which can reset these to defaults. Set them
    # again after init so the handler really uses a bounded worker pool.
    C.joblib_backend = cfg["runtime"]["joblib_backend"]
    C.kernels = int(cfg["runtime"]["kernels"])

    handler_kwargs = {
        "instruments": data_cfg["instruments"],
        "start_time": data_cfg["start_time"],
        "end_time": data_cfg["end_time"],
        "fit_start_time": data_cfg["fit_start_time"],
        "fit_end_time": data_cfg["fit_end_time"],
    }
    pit_membership = args.pit_membership or data_cfg.get("pit_membership")
    if pit_membership:
        from src.data.pit_handler import PITAlpha158

        membership_path = Path(pit_membership)
        if not membership_path.is_absolute():
            membership_path = PROJECT_ROOT / membership_path
        handler = PITAlpha158(membership_path=membership_path, **handler_kwargs)
        print(f"=== PIT membership: {membership_path} ===")
    elif mask:
        from src.data.masked_handler import MaskedAlpha158

        handler = MaskedAlpha158(
            exclude_dates=exclude_dates,
            exclude_instruments=exclude_instruments,
            **handler_kwargs,
        )
    else:
        handler = Alpha158(**handler_kwargs)
    dataset = DatasetH(handler, segments=segments)

    model_params = {
        k: v
        for k, v in model_cfg.items()
        if k not in {"num_boost_round", "early_stopping_rounds", "verbose_eval"}
    }
    model = LGBModel(
        loss="mse",
        num_boost_round=model_cfg["num_boost_round"],
        early_stopping_rounds=model_cfg["early_stopping_rounds"],
        seed=cfg["project"]["seed"],
        **model_params,
    )
    print("=== training ===")
    model.fit(
        dataset,
        num_boost_round=model_cfg["num_boost_round"],
        early_stopping_rounds=model_cfg["early_stopping_rounds"],
        verbose_eval=model_cfg["verbose_eval"],
    )

    print("=== predicting ===")
    pred_test = _normalize_series(model.predict(dataset, segment="test"), "pred")
    pred_valid = _normalize_series(model.predict(dataset, segment="valid"), "pred")
    if mask:
        pred_test = pred_test.loc[
            ~pred_test.index.get_level_values("datetime").isin(exclude_dates)
        ]
        pred_test = pred_test.loc[
            ~pred_test.index.get_level_values("instrument").isin(exclude_instruments)
        ]
        pred_valid = pred_valid.loc[
            ~pred_valid.index.get_level_values("datetime").isin(exclude_dates)
        ]
        pred_valid = pred_valid.loc[
            ~pred_valid.index.get_level_values("instrument").isin(exclude_instruments)
        ]

    test_start, test_end = segments["test"]
    label = D.features(
        D.instruments(data_cfg["instruments"]),
        ["Ref($close, -2) / Ref($close, -1) - 1"],
        start_time=test_start,
        end_time=test_end,
    )
    label.columns = ["label"]
    label = _normalize_series(label["label"], "label")

    eval_df = pd.concat([pred_test, label], axis=1, join="inner").dropna()
    if eval_df.empty:
        raise RuntimeError("Empty evaluation frame: check prediction/label index alignment.")
    eval_df.index.names = ["instrument", "datetime"]
    if mask:
        eval_df = eval_df.loc[~eval_df.index.get_level_values("datetime").isin(exclude_dates)]
        eval_df = eval_df.loc[~eval_df.index.get_level_values("instrument").isin(exclude_instruments)]

    rank_ic, pearson_ic = compute_ic(eval_df)
    qmean, long_short, qdf = quantile_stats(eval_df, q=5)
    turnover = top_quantile_turnover(eval_df, q=5)

    print("=== backtest ===")
    strategy = TopkDropoutStrategy(
        signal=pred_test,
        topk=bt_cfg["topk"],
        n_drop=bt_cfg["n_drop"],
        hold_thresh=bt_cfg["hold_thresh"],
        only_tradable=True,
        forbid_all_trade_at_limit=True,
    )
    executor = {
        "class": "SimulatorExecutor",
        "module_path": "qlib.backtest.executor",
        "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True},
    }
    exchange_kwargs = {
        "freq": "day",
        "limit_threshold": bt_cfg["limit_threshold"],
        "deal_price": bt_cfg["deal_price"],
        "open_cost": bt_cfg["open_cost"],
        "close_cost": bt_cfg["close_cost"],
        "min_cost": bt_cfg["min_cost"],
        "trade_unit": bt_cfg["trade_unit"],
    }
    portfolio_metric_dict, _indicator_dict = backtest(
        start_time=test_start,
        end_time=test_end,
        strategy=strategy,
        executor=executor,
        benchmark=data_cfg["benchmark"],
        account=bt_cfg["account"],
        exchange_kwargs=exchange_kwargs,
    )
    report_normal, _positions_normal = portfolio_metric_dict["1day"]
    report_normal_full = report_normal.copy()
    if mask:
        report_normal = report_normal.loc[~report_normal.index.isin(exclude_dates)]

    metrics: dict[str, Any] = {
        "project": cfg["project"],
        "data": data_cfg,
        "segments": segments,
        "model": model_cfg,
        "backtest_config": bt_cfg,
        "data_snapshot": {
            "id": snapshot.get("snapshot_id") if snapshot else None,
            "file": str(snapshot_path),
        },
        "pit_membership": str(membership_path) if pit_membership else None,
        "mask": {
            "file": args.mask_file,
            "excluded_dates": [str(d.date()) for d in exclude_dates],
            "excluded_instruments": sorted(exclude_instruments),
        }
        if mask
        else None,
        "test_rows": len(eval_df),
        "test_dates": int(rank_ic.size),
        "backtest_days": len(report_normal),
        "mean_cross_section_size": float(eval_df.groupby(level="datetime").size().mean()),
        "rank_ic": ic_stats(rank_ic),
        "pearson_ic": ic_stats(pearson_ic),
        "quantile": {
            "mean_forward_return": {str(k): float(v) for k, v in qmean.items()},
            "long_short_daily_mean": float(long_short.mean()) if not long_short.empty else None,
            "long_short_annualized_mean": float(long_short.mean() * 252) if not long_short.empty else None,
            "monotonicity_spearman": (
                float(qmean.corr(pd.Series(np.arange(len(qmean)), index=qmean.index), method="spearman"))
                if len(qmean) > 1
                else None
            ),
        },
        "top_quintile_turnover": turnover,
        "backtest": backtest_metrics(report_normal),
    }

    pred_test.to_frame("score").to_csv(out_dir / "pred_test.csv")
    pred_valid.to_frame("score").to_csv(out_dir / "pred_valid.csv")
    rank_ic.to_frame("rank_ic").to_csv(out_dir / "rank_ic.csv")
    pearson_ic.to_frame("ic").to_csv(out_dir / "ic.csv")
    qdf.to_csv(out_dir / "quantile_returns_by_date.csv")
    report_normal.to_csv(out_dir / "report_normal.csv")
    if mask:
        report_normal_full.to_csv(out_dir / "report_normal_full.csv")
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    (out_dir / "config_resolved.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    save_plots(rank_ic, qmean, report_normal, out_dir)

    print("=== summary ===")
    print(
        json.dumps(
            {
                "rank_ic": metrics["rank_ic"],
                "quantile": metrics["quantile"],
                "top_quintile_turnover": turnover,
                "backtest": {
                    k: v for k, v in metrics["backtest"].items()
                },
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )
    print(f"saved -> {out_dir}")


if __name__ == "__main__":
    main()
