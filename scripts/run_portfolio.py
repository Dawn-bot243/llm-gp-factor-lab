#!/usr/bin/env python
"""Stage 9: S3 portfolio construction, costs, constraints and attribution.

Protocol
--------
* factors: Stage 8 pass list (known + Stage 7 representatives);
* combination: equal / train-ICIR / LightGBM, fitted on 2010-2016 only;
* portfolio parameters: selected on 2017-2018 validation by net excess IR;
* test: 2019-2020-09-23, run once for the selected configuration;
* costs/constraints: explicit commission, stamp duty, transfer fee, slippage,
  T+1, suspension, ST, limit-up/down and ADV participation;
* reconciliation: Qlib TopkDropout reference with zero costs and no blocks.

Examples:
    .venv/Scripts/python.exe scripts/run_portfolio.py --quick --no-reconcile
    .venv/Scripts/python.exe scripts/run_portfolio.py
"""
from __future__ import annotations

import os

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import argparse
import copy
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.factors.evaluator import (
    DEFAULT_CONSTRAINTS,
    DEFAULT_SNAPSHOT,
    EvaluationConfig,
)
from src.factors.serialization import json_default, jsonable
from src.portfolio.attribution import (
    benchmark_exposure,
    brinson_attribution,
    exposure_summary,
    portfolio_exposure,
    size_bucket_groups,
    style_exposures,
)
from src.portfolio.combine import (
    build_combination,
    forward_returns,
    load_factor_panels,
    load_specs_from_stage8,
)
from src.portfolio.data import load_market_data, load_sector_map
from src.portfolio.metrics import selection_score
from src.portfolio.reconcile import compare_nav, qlib_topkdropout_report
from src.portfolio.simulator import PortfolioConfig, PortfolioResult, run_portfolio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "portfolio.yaml"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "artifacts" / "stage9"))
    parser.add_argument("--quick", action="store_true", help="small grid and shorter logs")
    parser.add_argument("--skip-grid", action="store_true", help="use the base topk/n_drop/hold")
    parser.add_argument("--no-reconcile", action="store_true")
    parser.add_argument("--methods", default=None, help="comma-separated subset of equal,icir,lgbm")
    parser.add_argument("--method", default=None, help="single combination method")
    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument("--n-drop", type=int, default=None)
    parser.add_argument("--hold-thresh", type=int, default=None)
    return parser.parse_args()


def _yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _merge(base: dict[str, Any], override: dict[str, Any] | None) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _run(
    scores: pd.DataFrame,
    config: PortfolioConfig,
    market,
    start: str,
    end: str,
    *,
    collect_weights: bool,
) -> PortfolioResult:
    return run_portfolio(
        scores=scores,
        open_px=market.open,
        close_px=market.close,
        amount_cny=market.amount_cny,
        benchmark_close=market.benchmark_close,
        config=config,
        start=start,
        end=end,
        suspended=market.suspended,
        is_st=market.is_st,
        at_limit_up=market.at_limit_up,
        at_limit_down=market.at_limit_down,
        factor=market.factor,
        collect_weights=collect_weights,
    )


def _row(
    *,
    method: str,
    topk: int,
    n_drop: int,
    hold_thresh: int,
    metrics: dict[str, Any],
    **extra: Any,
) -> dict[str, Any]:
    row = {
        "method": method,
        "topk": topk,
        "n_drop": n_drop,
        "hold_thresh": hold_thresh,
        "selection_score": selection_score(metrics),
        **extra,
    }
    row.update(metrics)
    return row


def main() -> int:
    args = parse_args()
    payload = _yaml(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline = _yaml(PROJECT_ROOT / payload["data"]["baseline_config"])
    data_cfg = baseline["data"]
    membership = data_cfg.get("pit_membership")
    if membership and not Path(membership).is_absolute():
        membership = str(PROJECT_ROOT / membership)
    data_config = EvaluationConfig(
        provider_uri=data_cfg["provider_uri"],
        instruments=data_cfg.get("instruments", "csi300"),
        start=payload["data"]["start"],
        end=payload["data"]["end"],
        membership_path=membership,
        constraints_path=str(DEFAULT_CONSTRAINTS),
        snapshot_file=str(DEFAULT_SNAPSHOT),
        use_cache=True,
        kernels=4,
    )
    print("=== loading factors ===")
    specs = load_specs_from_stage8(PROJECT_ROOT / payload["data"]["stage8_scorecard"], status=payload["data"]["stage8_status"])
    factors = load_factor_panels(
        specs,
        data_config,
        winsorize=payload["data"].get("winsorize", 0.01),
        standardize=bool(payload["data"].get("standardize", True)),
    )
    print(f"factors: {len(specs)} -> panel {factors.shape}")
    coverage = pd.DataFrame(
        {
            "coverage": factors.notna().mean(),
            "finite_rows": factors.notna().sum(),
            "n_dates": factors.notna().groupby(level="datetime").any().sum(),
        }
    )
    coverage.to_csv(output_dir / "factor_coverage.csv", index_label="name")

    print("=== loading market data ===")
    market = load_market_data(data_config)
    print(
        f"market: {len(market.dates)} dates x {len(market.instruments)} instruments; "
        f"benchmark {market.benchmark_close.index.min().date()}..{market.benchmark_close.index.max().date()}"
    )
    label = forward_returns(market.open, horizon=1, lag=1)
    print(f"label rows: {label.notna().sum():,}")

    if args.method:
        methods = [args.method]
    else:
        methods = [item.strip() for item in (args.methods or ",".join(payload["combination"]["methods"])).split(",") if item.strip()]
    lgbm_params = payload["combination"].get("lgbm", {})
    score_results = {}
    for method in methods:
        print(f"=== combination: {method} ===")
        result = build_combination(
            method,
            factors,
            label,
            train_start=payload["data"]["start"],
            train_end=payload["data"]["train_end"],
            min_names=int(payload["data"].get("min_names", 20)),
            min_dates=int(payload["data"].get("min_dates", 100)),
            lgbm_params=lgbm_params,
            seed=int(payload["project"].get("seed", 42)),
        )
        score_results[method] = result
        result.diagnostics.to_csv(output_dir / f"combination_{method}_diagnostics.csv", index_label="name")
        if result.feature_importance is not None:
            result.feature_importance.to_csv(output_dir / f"combination_{method}_importance.csv", index=False)
        print(
            f"{method}: score coverage {result.scores.notna().mean().mean():.3f}, "
            f"dates {result.scores.index.min().date()}..{result.scores.index.max().date()}"
        )

    base_portfolio = payload["portfolio"]
    grid = payload["grid"]
    if args.quick:
        grid = {"topk": [30, 50], "n_drop": [3, 5], "hold_thresh": [1, 5]}
    if args.skip_grid:
        grid = {
            "topk": [int(args.topk or base_portfolio.get("topk", 50))],
            "n_drop": [int(args.n_drop or base_portfolio.get("n_drop", 5))],
            "hold_thresh": [int(args.hold_thresh or base_portfolio.get("hold_thresh", 1))],
        }

    validation_rows: list[dict[str, Any]] = []
    print("=== validation grid ===")
    for method, score_result in score_results.items():
        for topk in grid["topk"]:
            for n_drop in grid["n_drop"]:
                for hold_thresh in grid["hold_thresh"]:
                    if n_drop > topk:
                        continue
                    portfolio_payload = _merge(
                        base_portfolio,
                        {"topk": int(topk), "n_drop": int(n_drop), "hold_thresh": int(hold_thresh)},
                    )
                    config = PortfolioConfig.from_dict(portfolio_payload)
                    simulation = _run(
                        score_result.scores,
                        config,
                        market,
                        payload["data"]["valid_start"],
                        payload["data"]["valid_end"],
                        collect_weights=False,
                    )
                    validation_rows.append(
                        _row(
                            method=method,
                            topk=int(topk),
                            n_drop=int(n_drop),
                            hold_thresh=int(hold_thresh),
                            metrics=simulation.metrics,
                        )
                    )
                    print(
                        f"{method} topk={topk} n_drop={n_drop} hold={hold_thresh}: "
                        f"excess_IR={simulation.metrics.get('net_excess_ir')}"
                    )
    validation = pd.DataFrame(validation_rows)
    validation = validation.sort_values(
        ["selection_score", "net_excess_ir", "turnover_annual"],
        ascending=[False, False, True],
        na_position="last",
    ).reset_index(drop=True)
    validation.to_csv(output_dir / "validation_grid.csv", index=False)

    if validation.empty:
        raise SystemExit("validation grid is empty")
    best_by_method = validation.sort_values("selection_score", ascending=False).groupby("method", as_index=False).first()
    best_by_method.to_csv(output_dir / "validation_best_by_method.csv", index=False)
    best = validation.iloc[0]
    selected_method = str(best["method"])
    selected = {
        "topk": int(best["topk"]),
        "n_drop": int(best["n_drop"]),
        "hold_thresh": int(best["hold_thresh"]),
    }
    print(f"selected: {selected_method} {selected}")
    print(best_by_method[["method", "topk", "n_drop", "hold_thresh", "net_excess_ir", "turnover_annual"]].to_string(index=False))

    primary_payload = _merge(base_portfolio, selected)
    primary_config = PortfolioConfig.from_dict(primary_payload)
    scenarios = {"net_base": {"description": "base costs and full constraints"}}
    scenarios.update(payload.get("scenarios", {}))
    scenario_rows: list[dict[str, Any]] = []
    primary_result: PortfolioResult | None = None
    print("=== test scenarios ===")
    for name, override in scenarios.items():
        scenario_payload = _merge(primary_payload, override)
        config = PortfolioConfig.from_dict(scenario_payload)
        simulation = _run(
            score_results[selected_method].scores,
            config,
            market,
            payload["data"]["test_start"],
            payload["data"]["end"],
            collect_weights=(name == "net_base"),
        )
        if name == "net_base":
            primary_result = simulation
            simulation.daily.to_csv(output_dir / "daily_nav.csv", index_label="datetime")
            if not simulation.trades.empty:
                simulation.trades.to_csv(output_dir / "trades.csv", index=False)
            if not simulation.weights.empty:
                simulation.weights.to_parquet(output_dir / "weights.parquet")
        scenario_rows.append(
            {
                "scenario": name,
                "description": override.get("description", ""),
                **simulation.metrics,
            }
        )
        print(
            f"{name}: net_ann={simulation.metrics.get('annual_return')} "
            f"excess_ann={simulation.metrics.get('excess_annual_return')} "
            f"IR={simulation.metrics.get('net_excess_ir')} "
            f"turnover={simulation.metrics.get('turnover_annual')}"
        )
    scenario_frame = pd.DataFrame(scenario_rows)
    scenario_frame.to_csv(output_dir / "test_scenarios.csv", index=False)

    if primary_result is None:
        raise SystemExit("primary test run failed")

    # Capacity: scale AUM with the same signal and execution rules.
    capacity_rows: list[dict[str, Any]] = []
    print("=== capacity ===")
    for scale in payload["capacity"]["aum_scales"]:
        account = float(base_portfolio["account"]) * float(scale)
        config = PortfolioConfig.from_dict(_merge(primary_payload, {"account": account}))
        simulation = _run(
            score_results[selected_method].scores,
            config,
            market,
            payload["data"]["test_start"],
            payload["data"]["end"],
            collect_weights=False,
        )
        capacity_rows.append(
            {
                "aum_scale": float(scale),
                "account": account,
                **simulation.metrics,
            }
        )
        print(
            f"AUM={account / 1e8:.2f}e8: excess_ann={simulation.metrics.get('excess_annual_return')} "
            f"IR={simulation.metrics.get('net_excess_ir')} fill={simulation.metrics.get('fill_rate_mean')}"
        )
    capacity = pd.DataFrame(capacity_rows)
    capacity.to_csv(output_dir / "capacity.csv", index=False)
    base_ir = capacity.loc[(capacity["aum_scale"] - 1.0).abs().idxmin(), "net_excess_ir"]
    fill_ok = capacity[
        capacity["fill_rate_mean"].fillna(0.0) >= float(payload["capacity"]["min_fill_rate"])
    ]
    if base_ir is not None and np.isfinite(base_ir) and float(base_ir) > 0:
        min_ir = float(base_ir) * float(payload["capacity"]["min_ir_retention"])
        eligible = fill_ok[fill_ok["net_excess_ir"].fillna(-1e9) >= min_ir]
        capacity_basis = "fill_rate_and_ir_retention"
    else:
        # If the strategy has no positive excess IR, an IR-retention rule is
        # meaningless.  Report an operational fill-rate capacity and say so.
        min_ir = None
        eligible = fill_ok
        capacity_basis = "fill_rate_only_ir_not_positive"
    capacity_estimate = {
        "base_ir": float(base_ir) if base_ir is not None and np.isfinite(base_ir) else None,
        "min_ir": min_ir,
        "min_fill_rate": float(payload["capacity"]["min_fill_rate"]),
        "basis": capacity_basis,
        "max_aum": float(eligible["account"].max()) if not eligible.empty else None,
        "max_aum_scale": float(eligible["aum_scale"].max()) if not eligible.empty else None,
    }

    # Attribution.
    print("=== attribution ===")
    returns = market.close / market.close.shift(1) - 1.0
    style = style_exposures(
        market.close,
        market.amount_cny,
        market.benchmark_close,
        min_names=int(payload["attribution"].get("min_names", 20)),
    )
    port_exposure = portfolio_exposure(primary_result.weights, style)
    bench_exposure = benchmark_exposure(style, market.membership_wide)
    exposure_summary(port_exposure, bench_exposure).to_csv(
        output_dir / "style_exposure_summary.csv", index=False
    )
    port_exposure.to_csv(output_dir / "style_exposure_portfolio.csv", index_label="datetime")
    bench_exposure.to_csv(output_dir / "style_exposure_benchmark.csv", index_label="datetime")

    sector_map = load_sector_map(payload["attribution"].get("sector_map"))
    membership_weights = market.membership_wide.div(
        market.membership_wide.sum(axis=1).replace(0, np.nan), axis=0
    )
    brinson_payload: dict[str, Any] = {}
    if not sector_map.empty:
        sector = brinson_attribution(
            primary_result.weights,
            returns,
            sector_map,
            benchmark_weights=membership_weights,
        )
        sector["daily"].to_csv(output_dir / "brinson_sector_daily.csv", index_label="datetime")
        sector["by_group"].to_csv(output_dir / "brinson_sector_by_group.csv", index=False)
        brinson_payload["sector"] = sector["summary"]
        print(f"sector Brinson: {sector['summary']}")
    size_groups = size_bucket_groups(
        style["size"],
        quantiles=int(payload["attribution"].get("size_quantiles", 5)),
    )
    size = brinson_attribution(
        primary_result.weights,
        returns,
        size_groups,
        benchmark_weights=membership_weights,
    )
    size["daily"].to_csv(output_dir / "brinson_size_daily.csv", index_label="datetime")
    size["by_group"].to_csv(output_dir / "brinson_size_by_group.csv", index=False)
    brinson_payload["size"] = size["summary"]
    print(f"size Brinson: {size['summary']}")

    # Reconciliation against Qlib's TopkDropoutStrategy.
    reconciliation: dict[str, Any] = {"enabled": not args.no_reconcile}
    if not args.no_reconcile:
        print("=== reconciliation: Qlib TopkDropout ===")
        try:
            qlib_config = PortfolioConfig.from_dict(
                _merge(
                    primary_payload,
                    {
                        "deal_price": "close",
                        "sizing": "qlib",
                        "only_tradable": False,
                        "qlib_compatible": True,
                        "costs": {
                            "enabled": False,
                        },
                        "constraints": {
                            "exclude_suspended": False,
                            "exclude_st": False,
                            "block_buy_at_limit_up": False,
                            "block_sell_at_limit_down": False,
                            "block_all_at_limit": False,
                            "t_plus_one": False,
                            "adv_participation": 0.0,
                        },
                    },
                )
            )
            qlib_sim = _run(
                score_results[selected_method].scores,
                qlib_config,
                market,
                payload["data"]["test_start"],
                payload["data"]["end"],
                collect_weights=False,
            )
            qlib_report = qlib_topkdropout_report(
                score_results[selected_method].scores,
                data_config,
                start=payload["data"]["test_start"],
                end=payload["data"]["end"],
                topk=primary_config.topk,
                n_drop=primary_config.n_drop,
                hold_thresh=primary_config.hold_thresh,
                account=primary_config.account,
                benchmark=data_cfg.get("benchmark", "SH000300"),
                risk_degree=primary_config.risk_degree,
            )
            reconciliation["qlib"] = compare_nav(qlib_sim.daily, qlib_report)
            reconciliation["qlib_config"] = qlib_config.to_dict()
            print(f"reconciliation: {reconciliation['qlib']}")
        except Exception as exc:  # pragma: no cover - engine/version dependent
            reconciliation["qlib"] = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"reconciliation failed: {reconciliation['qlib']}")

    summary = {
        "schema_version": "stage9.s3.v1",
        "generated_at": datetime.now(UTC).astimezone().isoformat(timespec="seconds"),
        "config": payload,
        "data": {
            "provider_uri": data_config.provider_uri,
            "start": data_config.start,
            "end": data_config.end,
            "snapshot_id": data_config.snapshot_id,
            "n_market_dates": len(market.dates),
            "n_instruments": len(market.instruments),
        },
        "factors": [spec.__dict__ for spec in specs],
        "factor_coverage": jsonable(coverage.reset_index().to_dict(orient="records")),
        "combination": {
            method: result.to_dict() for method, result in score_results.items()
        },
        "selected": {"method": selected_method, **selected},
        "validation": {
            "n_configs": len(validation),
            "n_configs_positive_ir": int((validation["net_excess_ir"] > 0).sum()),
            "selected_net_excess_ir": (
                float(best["net_excess_ir"])
                if best["net_excess_ir"] is not None and np.isfinite(best["net_excess_ir"])
                else None
            ),
            "validated": bool(
                best["net_excess_ir"] is not None
                and np.isfinite(best["net_excess_ir"])
                and float(best["net_excess_ir"]) > 0
            ),
        },
        "validation_best_by_method": jsonable(best_by_method.to_dict(orient="records")),
        "test_scenarios": jsonable(scenario_frame.to_dict(orient="records")),
        "capacity": {
            "table": jsonable(capacity.to_dict(orient="records")),
            "estimate": capacity_estimate,
        },
        "attribution": {"style": jsonable(port_exposure.describe().to_dict()), "brinson": jsonable(brinson_payload)},
        "reconciliation": jsonable(reconciliation),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(jsonable(summary), ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )
    print(
        json.dumps(
            jsonable(
                {
                    "selected": summary["selected"],
                    "test": {
                        row["scenario"]: {
                            "excess_annual_return": row.get("excess_annual_return"),
                            "net_excess_ir": row.get("net_excess_ir"),
                            "turnover_annual": row.get("turnover_annual"),
                        }
                        for row in summary["test_scenarios"]
                    },
                    "capacity": capacity_estimate,
                }
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
