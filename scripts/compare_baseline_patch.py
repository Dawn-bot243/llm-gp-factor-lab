#!/usr/bin/env python
"""Three-way comparison: original, pre-patch masked, post-patch masked."""
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
    parser.add_argument("--orig", default="artifacts/baseline/metrics.json")
    parser.add_argument("--pre-mask", default="artifacts/baseline_masked/metrics.json")
    parser.add_argument("--patched", default="artifacts/baseline_patched/metrics.json")
    parser.add_argument(
        "--patched-ex",
        default="artifacts/baseline_patched_ex_sz002493/metrics.json",
        help="optional patched run excluding reconstructed SZ002493",
    )
    parser.add_argument("--output-md", default="项目报告/baseline_patch_comparison.md")
    parser.add_argument("--output-json", default="项目报告/baseline_patch_comparison.json")
    args = parser.parse_args()

    runs = {
        "orig": json.loads(Path(args.orig).read_text(encoding="utf-8")),
        "pre_mask": json.loads(Path(args.pre_mask).read_text(encoding="utf-8")),
        "patched": json.loads(Path(args.patched).read_text(encoding="utf-8")),
    }
    if Path(args.patched_ex).exists():
        runs["patched_ex"] = json.loads(Path(args.patched_ex).read_text(encoding="utf-8"))

    def row(label: str, getter, digits: int = 4) -> str:
        values = [getter(metrics) for metrics in runs.values()]
        return "| " + label + " | " + " | ".join(_fmt(value, digits) for value in values) + " |"

    rows = [
        row("测试 IC 天数", lambda m: m.get("test_dates"), 0),
        row("测试样本行数", lambda m: m.get("test_rows"), 0),
        row("平均截面股票数", lambda m: m.get("mean_cross_section_size"), 2),
        row("RankIC 均值", lambda m: _get(m, "rank_ic", "mean"), 5),
        row("ICIR（日频）", lambda m: _get(m, "rank_ic", "icir"), 4),
        row("Newey-West t", lambda m: _get(m, "rank_ic", "newey_west_t"), 3),
        row("多空年化（毛）", lambda m: _get(m, "quantile", "long_short_annualized_mean"), 4),
        row("Top-20% 日换手", lambda m: m.get("top_quintile_turnover"), 4),
        row("组合年化收益", lambda m: _get(m, "backtest", "portfolio", "risk", "annualized_return"), 4),
        row("组合信息比率", lambda m: _get(m, "backtest", "portfolio", "risk", "information_ratio"), 3),
        row("组合最大回撤", lambda m: _get(m, "backtest", "portfolio", "risk", "max_drawdown"), 4),
        row("含成本超额年化", lambda m: _get(m, "backtest", "excess_with_cost", "risk", "annualized_return"), 4),
        row("含成本超额 IR", lambda m: _get(m, "backtest", "excess_with_cost", "risk", "information_ratio"), 3),
    ]

    has_ex = "patched_ex" in runs
    header = "| 指标 | 原始未掩码 | 补丁前掩码 | 补洞后（全 300） |"
    separator = "|---|---:|---:|---:|"
    if has_ex:
        header += " 补洞后（排 SZ002493） |"
        separator += "---:|"
    lines = [
        "# Baseline 对比：原始 / 掩码 / 补洞后",
        "",
        f"- 原始（未掩码）：`{args.orig}`",
        f"- 补丁前掩码：`{args.pre_mask}`",
        f"- 补洞后（全 300）：`{args.patched}`",
    ]
    if has_ex:
        lines.append(f"- 补洞后（排 SZ002493）：`{args.patched_ex}`")
    lines.extend(
        [
            "",
            header,
            separator,
            *rows,
            "",
            (
                "> 补洞版修复了 1,009 个短空洞标的-日和 1,564 个 SZ002493 长空洞标的-日；"
                "训练/验证/测试都使用修补后的数据，因此差异同时包含“补数据”和“重新训练”的影响，"
                "不能完全归因于评估期。"
            ),
            "",
            "## 结论",
            "",
            (
                "补洞后测试日从 418 回到 422。全 300 只版本：RankIC 0.0361、ICIR 0.279、"
                "含成本超额 23.5%、IR 2.40；排除 SZ002493 的敏感性版本为 23.0%、IR 2.35。"
                "两个版本都明显低于补丁前掩码的 26.0%、2.73。一个可见原因是 2019-04-29/30 "
                "恢复交易后，策略这两天分别跑输基准约 0.77pp 和 0.26pp；补丁前掩码把这两个"
                "缺失日排除了，结果偏乐观。由于训练集也被修补，无法把差异完全归因于评估期。"
                "补洞后的数字更诚实，应作为新的基线口径；SZ002493 的历史是 qfq 重建的，"
                "建议同时报告“排 SZ002493”的敏感性结果。"
            ),
            "",
        ]
    )
    md_path = Path(args.output_md)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines), encoding="utf-8")

    summary = {
        "orig": args.orig,
        "pre_mask": args.pre_mask,
        "patched": args.patched,
        "patched_ex": args.patched_ex if has_ex else None,
        "runs": {
            name: {
                "test_dates": m.get("test_dates"),
                "test_rows": m.get("test_rows"),
                "rank_ic_mean": _get(m, "rank_ic", "mean"),
                "icir": _get(m, "rank_ic", "icir"),
                "newey_west_t": _get(m, "rank_ic", "newey_west_t"),
                "long_short_annualized": _get(m, "quantile", "long_short_annualized_mean"),
                "turnover": m.get("top_quintile_turnover"),
                "excess_with_cost_annualized": _get(
                    m, "backtest", "excess_with_cost", "risk", "annualized_return"
                ),
                "excess_with_cost_ir": _get(m, "backtest", "excess_with_cost", "risk", "information_ratio"),
                "max_drawdown": _get(m, "backtest", "portfolio", "risk", "max_drawdown"),
            }
            for name, m in runs.items()
        },
    }
    json_path = Path(args.output_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved -> {md_path}")
    print(f"saved -> {json_path}")


if __name__ == "__main__":
    main()
