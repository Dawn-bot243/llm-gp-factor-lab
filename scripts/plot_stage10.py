#!/usr/bin/env python
"""Generate Stage 10 ablation and OOS charts."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = PROJECT_ROOT / "artifacts" / "stage10"
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
        "llm_gp_full": "D-full\n(2000)",
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
    axes[0].set_ylim(0, 1.35)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    axes[0].set_xticks(x, labels, rotation=0)

    axes[1].bar(x - width / 2, frame["s1_pass_rate"], width, color=BLUE, label="S1 pass")
    axes[1].bar(x + width / 2, frame["s1_accept_rate"], width, color=AQUA, label="S1 pass+review")
    axes[1].set_title("S1 gate")
    axes[1].set_ylim(0, 1.35)
    axes[1].legend(frameon=False, fontsize=8, loc="upper left")
    axes[1].set_xticks(x, labels, rotation=0)

    retention = pd.to_numeric(frame["oos_ic_retention"], errors="coerce")
    axes[2].bar(x, retention, color=[BLUE if value >= 0 else RED for value in retention.fillna(0.0)])
    axes[2].axhline(1.0, color=SECONDARY, linewidth=0.8, linestyle="--")
    axes[2].set_title("OOS IC retention")
    axes[2].set_ylim(0, max(1.4, float(retention.max(skipna=True) or 1.0) + 0.2))
    axes[2].set_xticks(x, labels, rotation=0)
    for index, value in enumerate(retention):
        if pd.notna(value):
            axes[2].text(index, value + 0.05, f"{value:.2f}", ha="center", fontsize=8)
        else:
            axes[2].text(index, 0.05, "n/a", ha="center", fontsize=8, color=SECONDARY)
    for axis in axes:
        axis.grid(axis="y", color=GRID, linewidth=0.8)
        axis.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(ASSETS / "stage10_search_quality.png", dpi=160)
    plt.close(fig)


def _s1_funnel(frame: pd.DataFrame) -> None:
    labels = _labels(frame)
    x = np.arange(len(frame))
    pass_values = frame["n_pass"].to_numpy(dtype=float)
    review_values = frame["n_review"].to_numpy(dtype=float)
    fail_values = frame["n_fail"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(9, 4.0))
    ax.bar(x, pass_values, color=BLUE, label="pass")
    ax.bar(x, review_values, bottom=pass_values, color=ORANGE, label="review")
    ax.bar(x, fail_values, bottom=pass_values + review_values, color=GRAY, label="fail")
    ax.set_ylabel("S1 candidates")
    ax.set_ylim(0, 6.8)
    ax.set_title("S1 pass / review / fail by ablation group")
    ax.set_xticks(x, labels)
    ax.legend(frameon=False, loc="upper right")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(ASSETS / "stage10_s1_funnel.png", dpi=160)
    plt.close(fig)


def _incremental(frame: pd.DataFrame) -> None:
    labels = _labels(frame)
    pre = pd.to_numeric(frame["incremental_pre_rank_ic"], errors="coerce").fillna(0.0)
    post = pd.to_numeric(frame["incremental_post_rank_ic"], errors="coerce").fillna(0.0)
    x = np.arange(len(frame))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 4.0))
    ax.bar(x - width / 2, pre, width, color=BLUE, label="pre-orthogonalisation")
    ax.bar(x + width / 2, post, width, color=AQUA, label="after Alpha158")
    ax.axhline(0.0, color=SECONDARY, linewidth=0.8)
    ax.set_ylabel("Mean RankIC")
    ax.set_title("Incremental IC after 20 Alpha158 exposures")
    ax.set_xticks(x, labels)
    ax.legend(frameon=False, loc="upper right")
    for index, (pre_value, post_value) in enumerate(zip(pre, post, strict=True)):
        if pre_value == 0 and post_value == 0:
            ax.text(index, 0.001, "n/a", ha="center", fontsize=8, color=SECONDARY)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(ASSETS / "stage10_incremental_ic.png", dpi=160)
    plt.close(fig)


def _oos(frame: pd.DataFrame) -> None:
    # D-full had zero holdings (its factors had no 2020-2023 coverage), so
    # cash-vs-negative-benchmark must not appear as a strategy result.
    frame = frame[frame["group"] != "llm_gp_full"].reset_index(drop=True)
    labels = _labels(frame)
    x = np.arange(len(frame))
    excess = pd.to_numeric(frame["oos_excess_annual"], errors="coerce")
    ir = pd.to_numeric(frame["oos_net_excess_ir"], errors="coerce")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))
    axes[0].bar(x, excess, color=[BLUE if value >= 0 else RED for value in excess.fillna(0.0)])
    axes[0].axhline(0.0, color=SECONDARY, linewidth=0.8)
    axes[0].set_title("Locked OOS net excess annual (D-full excluded)")
    axes[0].set_xticks(x, labels)
    for index, value in enumerate(excess):
        if pd.notna(value):
            axes[0].text(index, value + 0.004 if value >= 0 else value - 0.012, f"{value:.1%}", ha="center", fontsize=8)
    axes[1].bar(x, ir, color=[BLUE if value >= 0 else RED for value in ir.fillna(0.0)])
    axes[1].axhline(0.0, color=SECONDARY, linewidth=0.8)
    axes[1].set_title("Locked OOS net excess IR")
    axes[1].set_xticks(x, labels)
    for index, value in enumerate(ir):
        if pd.notna(value):
            axes[1].text(index, value + 0.03 if value >= 0 else value - 0.08, f"{value:.2f}", ha="center", fontsize=8)
    for axis in axes:
        axis.grid(axis="y", color=GRID, linewidth=0.8)
        axis.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(ASSETS / "stage10_oos.png", dpi=160)
    plt.close(fig)


def main() -> int:
    _style()
    ASSETS.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(ARTIFACTS / "ablation_summary.csv")
    _search_quality(frame)
    _s1_funnel(frame)
    _incremental(frame)
    _oos(frame)
    print(f"charts -> {ASSETS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
