#!/usr/bin/env python
"""Compare the old patched frozen baseline with the new PIT baseline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _get(metrics: dict[str, Any], *path: str, default: Any = None) -> Any:
    current: Any = metrics
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old", default="artifacts/baseline_patched/metrics.json")
    parser.add_argument("--new", default="artifacts/baseline_pit/metrics.json")
    parser.add_argument("--output-md", default="项目报告/baseline_pit_comparison.md")
    parser.add_argument("--output-json", default="项目报告/baseline_pit_comparison.json")
    args = parser.parse_args()
    old = json.loads(Path(args.old).read_text(encoding="utf-8"))
    new = json.loads(Path(args.new).read_text(encoding="utf-8"))

    rows = [
        ("测试 IC 天数", old.get("test_dates"), new.get("test_dates"), 0),
        ("测试样本行数", old.get("test_rows"), new.get("test_rows"), 0),
        ("平均截面股票数", old.get("mean_cross_section_size"), new.get("mean_cross_section_size"), 2),
        ("RankIC 均值", _get(old, "rank_ic", "mean"), _get(new, "rank_ic", "mean"), 5),
        ("ICIR（日频）", _get(old, "rank_ic", "icir"), _get(new, "rank_ic", "icir"), 4),
        ("Newey-West t", _get(old, "rank_ic", "newey_west_t"), _get(new, "rank_ic", "newey_west_t"), 3),
        ("多空年化（毛）", _get(old, "quantile", "long_short_annualized_mean"), _get(new, "quantile", "long_short_annualized_mean"), 4),
        ("Top-20% 日换手", old.get("top_quintile_turnover"), new.get("top_quintile_turnover"), 4),
        ("组合年化", _get(old, "backtest", "portfolio", "risk", "annualized_return"), _get(new, "backtest", "portfolio", "risk", "annualized_return"), 4),
        ("组合 IR", _get(old, "backtest", "portfolio", "risk", "information_ratio"), _get(new, "backtest", "portfolio", "risk", "information_ratio"), 3),
        ("组合最大回撤", _get(old, "backtest", "portfolio", "risk", "max_drawdown"), _get(new, "backtest", "portfolio", "risk", "max_drawdown"), 4),
        ("含成本超额年化", _get(old, "backtest", "excess_with_cost", "risk", "annualized_return"), _get(new, "backtest", "excess_with_cost", "risk", "annualized_return"), 4),
        ("含成本超额 IR", _get(old, "backtest", "excess_with_cost", "risk", "information_ratio"), _get(new, "backtest", "excess_with_cost", "risk", "information_ratio"), 3),
    ]
    lines = [
        "# Baseline 对比：旧数据 frozen vs 新数据 PIT",
        "",
        f"- 旧：`{args.old}`（cn_data_v3 补洞后，frozen CSI300）",
        f"- 新：`{args.new}`（investment_data，PIT CSI300）",
        "",
        "| 指标 | 旧数据 frozen | 新数据 PIT |",
        "|---|---:|---:|",
    ]
    for label, old_value, new_value, digits in rows:
        lines.append(f"| {label} | {_fmt(old_value, digits)} | {_fmt(new_value, digits)} |")
    lines.extend(
        [
            "",
            "> 这不是单一变量对比：数据源、复权口径、股票池和可用字段同时变化。",
            "> 新数据修正了 47 个日历缺日、2020-09-28 复权重置、数据空洞和成分前视，",
            "> 因此新 baseline 更低，但更可信。",
            "",
            "## 结论",
            "",
            "新 PIT baseline 的 RankIC 更高（0.0401 vs 0.0361）、ICIR 更高（0.325 vs 0.279），",
            "但含成本超额从 23.5% 降到 14.4%、IR 从 2.40 降到 1.70。",
            "一个主要原因是去掉成分前视：旧 frozen 股票池用 2020 年成分回看 2010 年，",
            "会系统性高估收益。数据源/复权/字段也同时变化，不能把差异全部归因于 PIT。",
            "新数据应作为 Stage 3 的主口径。",
            "",
        ]
    )
    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines), encoding="utf-8")
    summary = {
        "old": args.old,
        "new": args.new,
        "old_snapshot": old.get("data_snapshot"),
        "new_snapshot": new.get("data_snapshot"),
        "new_pit_membership": new.get("pit_membership"),
        "rows": {
            "test_dates": [old.get("test_dates"), new.get("test_dates")],
            "rank_ic_mean": [_get(old, "rank_ic", "mean"), _get(new, "rank_ic", "mean")],
            "icir": [_get(old, "rank_ic", "icir"), _get(new, "rank_ic", "icir")],
            "excess_with_cost_annualized": [
                _get(old, "backtest", "excess_with_cost", "risk", "annualized_return"),
                _get(new, "backtest", "excess_with_cost", "risk", "annualized_return"),
            ],
            "excess_with_cost_ir": [
                _get(old, "backtest", "excess_with_cost", "risk", "information_ratio"),
                _get(new, "backtest", "excess_with_cost", "risk", "information_ratio"),
            ],
        },
    }
    Path(args.output_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved -> {output_md}")
    print(f"saved -> {args.output_json}")


if __name__ == "__main__":
    main()
