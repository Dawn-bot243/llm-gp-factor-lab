#!/usr/bin/env python
"""Render Stage 10C result tables as Markdown for the project report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LABELS = {
    "random": "A random",
    "gp_only": "B GP-only",
    "llm_only": "C LLM-only",
    "llm_gp": "D LLM+GP",
}


def pct(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None else f"{float(value):+.{digits}%}"


def num(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", default=str(PROJECT_ROOT / "artifacts" / "stage10c"))
    parser.add_argument("--stage10b", default=str(PROJECT_ROOT / "artifacts" / "stage10b"))
    parser.add_argument(
        "--output", default=str(PROJECT_ROOT / "artifacts" / "stage10c" / "report_tables.md")
    )
    args = parser.parse_args()

    artifacts = Path(args.artifacts)
    summary = json.loads((artifacts / "summary.json").read_text(encoding="utf-8"))
    rows = {item["group"]: item for item in summary["ablation"]}

    lines: list[str] = []
    lines += [
        "### 4.1 S1",
        "",
        "| 组 | returned | unique | eligible | selected | S1 | pass / review / fail / error | pass rate | 95% CI | 校准 RankIC | OOS RankIC |",
        "|---|---:|---:|---:|---:|---:|---|---:|---|---:|---:|",
    ]
    for group, item in rows.items():
        lines.append(
            f"| {LABELS.get(group, group)} | {item['n_returned']} | {item['n_unique']} | "
            f"{item['n_eligible']} | {item['n_selected']} | {item['n_s1']} | "
            f"{item['n_pass']} / {item['n_review']} / {item['n_fail']} / {item['n_error']} | "
            f"{pct(item['s1_pass_rate'], 1)} | "
            f"[{pct(item['s1_ci_low'], 1)}, {pct(item['s1_ci_high'], 1)}] | "
            f"{num(item['mean_s1_rank_ic'], 4)} | {num(item['mean_oos_rank_ic'], 4)} |"
        )

    lines += [
        "",
        "### 4.2 增量 IC",
        "",
        "| 组 | 因子数 | 正交化前 RankIC | 正交化后 RankIC | 降幅 |",
        "|---|---:|---:|---:|---:|",
    ]
    for group, item in rows.items():
        lines.append(
            f"| {LABELS.get(group, group)} | {item['incremental_n']} | "
            f"{num(item['incremental_pre_rank_ic'], 4)} | "
            f"{num(item['incremental_post_rank_ic'], 4)} | "
            f"{abs(float(item['incremental_ic_drop'])):.4f} |"
        )

    lines += [
        "",
        "### 4.3 OOS 组合",
        "",
        "| 组 | 因子数 | 净年化 | 净超额 | Sharpe | 净超额 IR | 年换手 | 最大回撤 | 成本拖累 | OOS 组合 RankIC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group, item in rows.items():
        lines.append(
            f"| {LABELS.get(group, group)} | {item.get('n_s3_factors')} | "
            f"{pct(item.get('oos_net_annual'))} | {pct(item.get('oos_excess_annual'))} | "
            f"{num(item.get('oos_sharpe'))} | {num(item.get('oos_net_excess_ir'))} | "
            f"{num(item.get('oos_turnover_annual'), 2)}x | "
            f"{pct(item.get('oos_max_drawdown'))} | "
            f"{pct(item.get('oos_cost_drag_annual'))} | "
            f"{num(item.get('oos_rank_ic'), 4)} |"
        )

    bootstrap_path = artifacts / "bootstrap_oos.json"
    if bootstrap_path.exists():
        boot = json.loads(bootstrap_path.read_text(encoding="utf-8"))
        lines += [
            "",
            "### 4.4 Block bootstrap（20 日块，2000 次）",
            "",
            "| 组 | 年化净超额 | 95% CI | 净超额 IR | 95% CI |",
            "|---|---:|---|---:|---|",
        ]
        for group, item in boot.get("groups", {}).items():
            lines.append(
                f"| {item['label']} | {pct(item['annual_excess'])} | "
                f"[{pct(item['annual_excess_ci_low'])}, {pct(item['annual_excess_ci_high'])}] | "
                f"{num(item['net_excess_ir'])} | "
                f"[{num(item['net_excess_ir_ci_low'])}, {num(item['net_excess_ir_ci_high'])}] |"
            )
        for name, item in boot.get("differences", {}).items():
            lines.append(
                f"| {name}（差值） | {pct(item['annual_excess_diff'])} | "
                f"[{pct(item['annual_excess_diff_ci_low'])}, {pct(item['annual_excess_diff_ci_high'])}] | "
                f"{num(item['ir_diff'])} | "
                f"bootstrap 正值占比 {item['share_boot_positive']:.1%} |"
            )

    comparison_path = artifacts / "s1_reproduction.json"
    if comparison_path.exists():
        comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
        lines += [
            "",
            "### 5.1 S1 复现（10B vs 10C）",
            "",
            "| 组 | 10B pass | 10C pass | 状态变化 | pass 丢失 | pass 新增 | max |Δcoverage| |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for group, item in comparison.get("groups", {}).items():
            if item.get("status") == "missing":
                continue
            lines.append(
                f"| {LABELS.get(group, group)} | {item['pass_before']} | {item['pass_after']} | "
                f"{item['n_status_changed']} | {item['pass_lost']} | {item['pass_gained']} | "
                f"{item['max_abs_coverage_diff']:.4f} |"
            )

    stage10b_path = Path(args.stage10b) / "summary.json"
    if stage10b_path.exists():
        before = json.loads(stage10b_path.read_text(encoding="utf-8"))
        before_rows = {item["group"]: item for item in before["ablation"]}
        lines += [
            "",
            "### 5.2 OOS 对比（10B 2020–2023 vs 10C 2023–2026）",
            "",
            "| 组 | 10B 净超额 | 10B IR | 10C 净超额 | 10C IR | 变化（pp） |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for group, item in rows.items():
            old = before_rows.get(group, {})
            old_excess = old.get("oos_excess_annual")
            new_excess = item.get("oos_excess_annual")
            change = (
                f"{(float(new_excess) - float(old_excess)) * 100:+.2f}"
                if old_excess is not None and new_excess is not None
                else "n/a"
            )
            lines.append(
                f"| {LABELS.get(group, group)} | {pct(old_excess)} | {num(old.get('oos_net_excess_ir'))} | "
                f"{pct(new_excess)} | {num(item.get('oos_net_excess_ir'))} | {change} |"
            )

    text = "\n".join(lines) + "\n"
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
