#!/usr/bin/env python
"""Generate Stage 9 report charts from artifacts/stage9.

The charts follow the project's validated light-mode palette:
blue = strategy, orange = benchmark, red/blue diverging for signed heatmaps.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = PROJECT_ROOT / "artifacts" / "stage9"
ASSETS = PROJECT_ROOT / "项目报告" / "assets"

BLUE = "#2a78d6"
ORANGE = "#eb6834"
RED = "#d03b3b"
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
SECONDARY = "#52514e"
GRID = "#e5e4df"
NEUTRAL = "#f0efec"


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
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
        }
    )


def _nav_chart(daily: pd.DataFrame) -> None:
    nav = daily["nav"].astype(float)
    nav = nav / nav.iloc[0]
    bench = (1.0 + daily["benchmark_return"].fillna(0.0)).cumprod()
    bench = bench / bench.iloc[0]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(nav.index, nav.values, color=BLUE, linewidth=2.0, label="Strategy (net)")
    ax.plot(bench.index, bench.values, color=ORANGE, linewidth=2.0, label="CSI300")
    ax.set_title("Stage 9 net NAV vs CSI300, 2019-2020")
    ax.set_ylabel("Cumulative NAV (start = 1)")
    ax.legend(loc="upper left", frameon=False)
    ax.annotate(
        f"{nav.iloc[-1]:.3f}",
        xy=(nav.index[-1], nav.iloc[-1]),
        xytext=(-4, 8),
        textcoords="offset points",
        color=BLUE,
        ha="right",
        fontsize=9,
    )
    ax.annotate(
        f"{bench.iloc[-1]:.3f}",
        xy=(bench.index[-1], bench.iloc[-1]),
        xytext=(-4, -14),
        textcoords="offset points",
        color=ORANGE,
        ha="right",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(ASSETS / "stage9_nav.png", dpi=160)
    plt.close(fig)


def _drawdown_chart(daily: pd.DataFrame) -> None:
    nav = daily["nav"].astype(float)
    nav = nav / nav.iloc[0]
    dd = nav / nav.cummax() - 1.0
    fig, ax = plt.subplots(figsize=(9, 3.0))
    ax.fill_between(dd.index, dd.values, 0.0, color=BLUE, alpha=0.18, linewidth=0)
    ax.plot(dd.index, dd.values, color=BLUE, linewidth=1.8)
    ax.axhline(0.0, color=SECONDARY, linewidth=0.8)
    ax.set_title("Stage 9 strategy drawdown (net)")
    ax.set_ylabel("Drawdown")
    fig.tight_layout()
    fig.savefig(ASSETS / "stage9_drawdown.png", dpi=160)
    plt.close(fig)


def _validation_heatmap(validation: pd.DataFrame, selected: dict) -> None:
    method = selected["method"]
    data = validation[validation["method"] == method]
    holds = sorted(data["hold_thresh"].unique())
    topks = sorted(data["topk"].unique())
    ndrops = sorted(data["n_drop"].unique())
    values = data["net_excess_ir"].to_numpy(dtype=float)
    limit = float(np.nanmax(np.abs(values))) if np.isfinite(values).any() else 1.0
    cmap = LinearSegmentedColormap.from_list("stage9_div", [BLUE, NEUTRAL, RED])
    fig, axes = plt.subplots(
        1, len(holds), figsize=(4.0 * len(holds), 3.4), sharey=True, constrained_layout=True
    )
    if len(holds) == 1:
        axes = [axes]
    for ax, hold in zip(axes, holds, strict=True):
        subset = data[data["hold_thresh"] == hold]
        pivot = subset.pivot(index="topk", columns="n_drop", values="net_excess_ir").reindex(
            index=topks, columns=ndrops
        )
        image = ax.imshow(
            pivot.to_numpy(dtype=float),
            cmap=cmap,
            vmin=-limit,
            vmax=limit,
            aspect="auto",
        )
        ax.set_xticks(range(len(ndrops)), [str(value) for value in ndrops])
        ax.set_yticks(range(len(topks)), [str(value) for value in topks])
        ax.set_xlabel("n_drop")
        ax.set_title(f"hold_thresh={hold}")
        ax.grid(False)
        for row in range(len(topks)):
            for col in range(len(ndrops)):
                value = pivot.iloc[row, col]
                if pd.notna(value):
                    ax.text(
                        col,
                        row,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color=TEXT if abs(value) < limit * 0.6 else SURFACE,
                    )
    axes[0].set_ylabel("topk")
    fig.suptitle(
        f"Validation net excess IR by parameters ({method}); selected = "
        f"topk {selected['topk']}, n_drop {selected['n_drop']}, hold {selected['hold_thresh']}",
        fontsize=11,
        fontweight="bold",
    )
    fig.colorbar(image, ax=axes, shrink=0.8, label="net excess IR")
    fig.savefig(ASSETS / "stage9_validation_heatmap.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _capacity_chart(capacity: pd.DataFrame) -> None:
    aum = capacity["account"] / 1e8
    fig, axes = plt.subplots(2, 1, figsize=(8, 5.2), sharex=True)
    axes[0].plot(aum, capacity["excess_annual_return"], color=BLUE, marker="o", linewidth=2.0)
    axes[0].axhline(0.0, color=SECONDARY, linewidth=0.8)
    axes[0].set_ylabel("Net excess annual")
    axes[0].set_title("Capacity: net excess return and fill rate vs AUM")
    axes[1].plot(aum, capacity["fill_rate_mean"], color=BLUE, marker="o", linewidth=2.0)
    axes[1].axhline(0.95, color=ORANGE, linewidth=1.0, linestyle="--")
    axes[1].set_ylabel("Mean fill rate")
    axes[1].set_xlabel("AUM (CNY 100m, log scale)")
    axes[1].set_xscale("log")
    axes[1].set_ylim(0.90, 1.005)
    for axis in axes:
        axis.grid(True, which="major", color=GRID, linewidth=0.8)
    fig.tight_layout()
    fig.savefig(ASSETS / "stage9_capacity.png", dpi=160)
    plt.close(fig)


def _scenario_chart(scenarios: pd.DataFrame) -> None:
    frame = scenarios.copy()
    frame = frame.sort_values("excess_annual_return")
    labels = frame["scenario"].astype(str).tolist()
    values = frame["excess_annual_return"].astype(float).to_numpy()
    colors = [RED if value < 0 else BLUE for value in values]
    fig, ax = plt.subplots(figsize=(8, 4.0))
    bars = ax.barh(labels, values, color=colors, height=0.55)
    ax.axvline(0.0, color=SECONDARY, linewidth=0.8)
    ax.set_xlabel("Net excess annual return")
    ax.set_title("Stage 9 test scenarios: net excess return")
    ax.set_xlim(float(values.min()) - 0.06, 0.02)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    for bar, value in zip(bars, values, strict=True):
        offset = 0.004 if value >= 0 else -0.004
        ax.text(
            value + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1%}",
            va="center",
            ha="left" if value >= 0 else "right",
            color=TEXT,
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(ASSETS / "stage9_scenarios.png", dpi=160)
    plt.close(fig)


def main() -> int:
    _style()
    ASSETS.mkdir(parents=True, exist_ok=True)
    daily = pd.read_csv(ARTIFACTS / "daily_nav.csv", parse_dates=["datetime"], index_col="datetime")
    validation = pd.read_csv(ARTIFACTS / "validation_grid.csv")
    capacity = pd.read_csv(ARTIFACTS / "capacity.csv")
    scenarios = pd.read_csv(ARTIFACTS / "test_scenarios.csv")
    import json

    selected = json.loads((ARTIFACTS / "summary.json").read_text(encoding="utf-8"))["selected"]
    _nav_chart(daily)
    _drawdown_chart(daily)
    _validation_heatmap(validation, selected)
    _capacity_chart(capacity)
    _scenario_chart(scenarios)
    print(f"charts -> {ASSETS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
