#!/usr/bin/env python
"""Generate Stage 10B ablation and OOS charts."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = PROJECT_ROOT / "artifacts" / "stage10b"
ASSETS = PROJECT_ROOT / "项目报告" / "assets"

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
RED = "#d03b3b"
GRAY = "#8a8984"
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
SECONDARY = "#52514e"
GRID = "#e5e4df"


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "text.color": TEXT,
            "axes.labelcolor": TEXT,
            "xtick.color": SECONDARY,
            "ytick.color": SECONDARY,
            "axes.edgecolor": GRID,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
        }
    )


def _labels(frame: pd.DataFrame) -> list[str]:
    mapping = {
        "random": "A\nrandom",
        "gp_only": "B\nGP-only",
        "llm_only": "C\nLLM-only",
        "llm_gp": "D\nLLM+GP",
    }
    return [mapping.get(str(group), str(group)) for group in frame["group"]]


def _search_quality(frame: pd.DataFrame) -> None:
    labels = _labels(frame)
    x = np.arange(len(frame))
    width = 0.36
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    axes[0].bar(x - width / 2, frame["legal_rate"], width, color=BLUE, label="legal / returned")
    axes[0].bar(x + width / 2, frame["unique_rate"], width, color=ORANGE, label="unique / returned")
    axes[0].set_title("Search legality")
    axes[0].set_ylim(0, 1.15)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    axes[0].set_xticks(x, labels, rotation=0)

    pass_rate = pd.to_numeric(frame["s1_pass_rate"], errors="coerce")
    ci_low = pd.to_numeric(frame["s1_ci_low"], errors="coerce")
    ci_high = pd.to_numeric(frame["s1_ci_high"], errors="coerce")
    errors = np.vstack([pass_rate - ci_low, ci_high - pass_rate])
    axes[1].bar(x, pass_rate, 0.55, color=BLUE, label="S1 pass rate")
    axes[1].errorbar(x, pass_rate, yerr=errors, fmt="none", ecolor=SECONDARY, capsize=4, linewidth=1.0)
    axes[1].set_title("S1 pass rate (95% Clopper-Pearson CI)")
    axes[1].set_ylim(0, 1.0)
    axes[1].set_xticks(x, labels, rotation=0)
    for index, (value, low, high) in enumerate(zip(pass_rate, ci_low, ci_high, strict=True)):
        if pd.notna(value):
            axes[1].text(index, high + 0.03, f"{value:.1%}", ha="center", fontsize=8)

    calibration = pd.to_numeric(frame["mean_s1_rank_ic"], errors="coerce")
    oos = pd.to_numeric(frame["mean_oos_rank_ic"], errors="coerce")
    axes[2].bar(x - width / 2, calibration, width, color=BLUE, label="calibration 2010-2016")
    axes[2].bar(x + width / 2, oos, width, color=AQUA, label="OOS 2020-2023")
    axes[2].axhline(0.0, color=SECONDARY, linewidth=0.8)
    axes[2].set_title("Mean factor RankIC")
    axes[2].legend(frameon=False, fontsize=8, loc="lower right")
    axes[2].set_xticks(x, labels, rotation=0)

    for axis in axes:
        axis.grid(axis="y", color=GRID, linewidth=0.8)
        axis.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(ASSETS / "stage10b_search_quality.png", dpi=160)
    plt.close(fig)


def _s1_funnel(frame: pd.DataFrame) -> None:
    labels = _labels(frame)
    x = np.arange(len(frame))
    pass_values = frame["n_pass"].to_numpy(dtype=float)
    review_values = frame["n_review"].to_numpy(dtype=float)
    fail_values = frame["n_fail"].to_numpy(dtype=float)
    error_values = frame["n_error"].to_numpy(dtype=float)
    totals = pass_values + review_values + fail_values + error_values

    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.bar(x, pass_values, color=BLUE, label="pass")
    bottom = pass_values
    ax.bar(x, review_values, bottom=bottom, color=ORANGE, label="review")
    bottom = bottom + review_values
    ax.bar(x, fail_values, bottom=bottom, color=GRAY, label="fail")
    bottom = bottom + fail_values
    ax.bar(x, error_values, bottom=bottom, color=RED, label="error (timeout)")
    ax.set_ylabel("S1 candidates")
    ax.set_ylim(0, totals.max() * 1.22)
    ax.set_title("S1 pass / review / fail / error by ablation group", pad=26)
    ax.set_xticks(x, labels)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=4)
    for index, (total, n_pass) in enumerate(zip(totals, pass_values, strict=True)):
        ax.text(index, total + 0.5, f"n={int(total)}, pass={int(n_pass)}", ha="center", fontsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(ASSETS / "stage10b_s1_funnel.png", dpi=160)
    plt.close(fig)


def _incremental(frame: pd.DataFrame) -> None:
    labels = _labels(frame)
    pre = pd.to_numeric(frame["incremental_pre_rank_ic"], errors="coerce").fillna(0.0)
    post = pd.to_numeric(frame["incremental_post_rank_ic"], errors="coerce").fillna(0.0)
    x = np.arange(len(frame))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 4.0))
    ax.bar(x - width / 2, pre, width, color=BLUE, label="pre-orthogonalisation")
    ax.bar(x + width / 2, post, width, color=AQUA, label="after 20 Alpha158 exposures")
    ax.axhline(0.0, color=SECONDARY, linewidth=0.8)
    ax.set_ylabel("Mean RankIC")
    ax.set_title("Incremental IC after 20 Alpha158 exposures (2010-2016)")
    ax.set_xticks(x, labels)
    ax.legend(frameon=False, loc="lower right")
    for index, (pre_value, post_value) in enumerate(zip(pre, post, strict=True)):
        if pre_value == 0 and post_value == 0:
            ax.text(index, 0.001, "n/a", ha="center", fontsize=8, color=SECONDARY)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(ASSETS / "stage10b_incremental_ic.png", dpi=160)
    plt.close(fig)


def _oos(frame: pd.DataFrame) -> None:
    labels = _labels(frame)
    x = np.arange(len(frame))
    excess = pd.to_numeric(frame["oos_excess_annual"], errors="coerce")
    ir = pd.to_numeric(frame["oos_net_excess_ir"], errors="coerce")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))
    axes[0].bar(x, excess, color=[BLUE if value >= 0 else RED for value in excess.fillna(0.0)])
    axes[0].axhline(0.0, color=SECONDARY, linewidth=0.8)
    axes[0].set_title("Exploratory OOS net excess annual")
    axes[0].set_xticks(x, labels)
    for index, value in enumerate(excess):
        if pd.notna(value):
            offset = 0.004 if value >= 0 else -0.012
            axes[0].text(index, value + offset, f"{value:+.2%}", ha="center", fontsize=8)
    axes[1].bar(x, ir, color=[BLUE if value >= 0 else RED for value in ir.fillna(0.0)])
    axes[1].axhline(0.0, color=SECONDARY, linewidth=0.8)
    axes[1].set_title("Exploratory OOS net excess IR")
    axes[1].set_xticks(x, labels)
    for index, value in enumerate(ir):
        if pd.notna(value):
            offset = 0.03 if value >= 0 else -0.08
            axes[1].text(index, value + offset, f"{value:.2f}", ha="center", fontsize=8)
    for axis in axes:
        low, high = axis.get_ylim()
        pad = (high - low) * 0.15
        axis.set_ylim(low - pad, high + pad)
        axis.grid(axis="y", color=GRID, linewidth=0.8)
        axis.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(ASSETS / "stage10b_oos.png", dpi=160)
    plt.close(fig)


def _sensitivity() -> None:
    frame = pd.read_csv(ARTIFACTS / "s1_sensitivity.csv")
    frame = frame[frame["coverage_threshold"] == 0.80]
    groups = list(dict.fromkeys(frame["group"]))
    mapping = {"random": "A random", "gp_only": "B GP-only", "llm_only": "C LLM-only", "llm_gp": "D LLM+GP"}
    colors = {"random": BLUE, "gp_only": ORANGE, "llm_only": AQUA, "llm_gp": RED}
    offsets = {"random": 0.015, "gp_only": 0.015, "llm_only": 0.015, "llm_gp": -0.04}
    thresholds = sorted(frame["ic_threshold"].unique())
    positions = {value: index for index, value in enumerate(thresholds)}

    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    for group in groups:
        subset = frame[frame["group"] == group].sort_values("ic_threshold")
        x = [positions[value] for value in subset["ic_threshold"]]
        ax.plot(x, subset["pass_rate"], marker="o", linewidth=1.8, color=colors.get(group, GRAY), label=mapping.get(group, group))
        for x_value, y_value in zip(x, subset["pass_rate"], strict=True):
            ax.text(x_value, y_value + offsets.get(group, 0.012), f"{y_value:.0%}", ha="center", fontsize=8)
    ax.set_xticks(list(positions.values()), [f"{value:.4g}" for value in thresholds])
    ax.set_xlabel("|RankIC| threshold")
    ax.set_ylabel("S1 pass rate")
    ax.set_ylim(0, 0.65)
    ax.set_title("S1 pass-rate sensitivity to IC threshold (coverage 0.80)")
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(ASSETS / "stage10b_s1_sensitivity.png", dpi=160)
    plt.close(fig)


def main() -> int:
    _style()
    ASSETS.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(ARTIFACTS / "ablation_summary.csv")
    _search_quality(frame)
    _s1_funnel(frame)
    _incremental(frame)
    _oos(frame)
    _sensitivity()
    print(f"charts -> {ASSETS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
