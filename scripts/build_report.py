#!/usr/bin/env python
"""Build a Markdown project report from Stage 0/1 artifacts."""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = PROJECT_ROOT / "artifacts" / "baseline"
REPORT_DIR = PROJECT_ROOT / "项目报告"
ASSETS_DIR = REPORT_DIR / "assets"


def pct(x, digits=2):
    if x is None:
        return "N/A"
    return f"{100.0 * float(x):.{digits}f}%"


def num(x, digits=4):
    if x is None:
        return "N/A"
    return f"{float(x):.{digits}f}"


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"missing artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def risk_row(backtest: dict, name: str) -> dict:
    return backtest.get(name, {}).get("risk", {})


def main() -> None:
    metrics = load_json(ARTIFACTS / "metrics.json")
    data = metrics["data"]
    segments = metrics["segments"]
    model = metrics["model"]
    rank = metrics["rank_ic"]
    pearson = metrics["pearson_ic"]
    quant = metrics["quantile"]
    portfolio = risk_row(metrics["backtest"], "portfolio")
    excess_no_cost = risk_row(metrics["backtest"], "excess_without_cost")
    excess_cost = risk_row(metrics["backtest"], "excess_with_cost")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    for png in [
        "cumulative_rank_ic.png",
        "quantile_returns.png",
        "cumulative_return.png",
        "drawdown.png",
    ]:
        src = ARTIFACTS / png
        if src.exists():
            shutil.copy2(src, ASSETS_DIR / png)

    coverage = {}
    coverage_path = PROJECT_ROOT / "项目报告" / "data_coverage.json"
    if coverage_path.exists():
        coverage = load_json(coverage_path)
    quality = {}
    quality_path = PROJECT_ROOT / "项目报告" / "data_quality.json"
    if quality_path.exists():
        quality = load_json(quality_path)

    lines: list[str] = []
    lines.append("# LLM + GP 因子挖掘项目报告：Stage 0 + Stage 1")
    lines.append("")
    lines.append(f"> 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ")
    lines.append("> 项目路径：`D:/quant/llm-gp-factor-lab`  ")
    lines.append("> 报告目录已加入 `.gitignore`，不会上传到 GitHub。")
    lines.append("")
    lines.append("## 1. 本阶段目标")
    lines.append("")
    lines.append("本阶段只做两件事，不涉及 LLM/GP：")
    lines.append("")
    lines.append("1. **Stage 0**：搭好 Windows 原生 Python 3.12 + Qlib 环境，准备 A 股日频数据，验证数据可读。")
    lines.append("2. **Stage 1**：跑通 `Alpha158 + LightGBM + TopkDropout` 基线，产出因子级和组合级指标。")
    lines.append("")
    lines.append("这条基线是后续 LLM/GP 因子挖掘的参照物。没有它，后续“增量”无法被证明。")
    lines.append("")
    lines.append("## 2. 环境与数据")
    lines.append("")
    lines.append("| 项目 | 值 |")
    lines.append("|---|---|")
    lines.append("| 操作系统 | Windows 11 |")
    lines.append("| Python | 3.12.3 |")
    lines.append("| Qlib | 0.9.7（Windows cp312 wheel） |")
    lines.append("| LightGBM | 4.7.0 |")
    lines.append("| 数据目录 | `D:/qlib_data/cn_data_v3` |")
    lines.append("| 原始覆盖 | 2010-01-04 至 2022-12-30 |")
    lines.append(f"| **实际使用区间** | **{data['start_time']} 至 {data['end_time']}** |")
    lines.append("| 交易日数 | 3112（原始覆盖） |")
    lines.append("| 全市场标的数 | 4368 |")
    lines.append("| 本次股票池 | `csi300_frozen`（300 只） |")
    lines.append("")
    if coverage:
        lines.append("数据覆盖检查（`scripts/check_data.py`）：")
        lines.append("")
        lines.append("| 指标 | 值 |")
        lines.append("|---|---|")
        lines.append(f"| 日历起点 | {coverage.get('calendar_start')} |")
        lines.append(f"| 日历终点 | {coverage.get('calendar_end')} |")
        lines.append(f"| 交易日 | {coverage.get('calendar_days')} |")
        lines.append(f"| 全市场标的 | {coverage.get('all_instruments')} |")
        lines.append(f"| csi300 历史标的 | {coverage.get('csi300_instruments')} |")
        lines.append(f"| 原始字段 | {', '.join(coverage.get('fields', []))} |")
        lines.append("")
    lines.append("### 2.1 股票池说明")
    lines.append("")
    lines.append("Qlib 自带的 `csi300.txt` 成分文件截止 **2020-09-25**，直接使用会让 2020-09-25 之后的测试集为空。")
    lines.append("因此本次构造了 `csi300_frozen`（重建脚本 `scripts/build_universe.py`）：")
    lines.append("")
    lines.append("- 取 2020-09-25 当期的 300 只 csi300 成分；")
    lines.append("- 用 `all.txt` 中的真实起止日期替换，保留 1 只覆盖到 2020-09-25 的退出样本；")
    lines.append("- 299 只覆盖到 2022-12-30，1 只覆盖到 2020-09-25。")
    lines.append("")
    lines.append("**偏差**：股票池使用 2020 年成分信息，对 2010–2016 的训练期存在前视；")
    lines.append("但它不是“只保留 2022 年还活着”的强幸存者偏差。Stage 2 应换成 point-in-time 成分。")
    lines.append("")
    lines.append("### 2.2 数据已知问题")
    lines.append("")
    lines.append("- Qlib `cn_data_v3` 来自 Yahoo Finance 社区数据，**不是 PIT 基本面库**；")
    lines.append("- **2020-09-28 存在价格尺度断裂**：复权收盘价整体重置到 ~1.0，成交量跳变约 10 倍，基准单日收益 −78%。该日期之后的数据不可直接使用；本次 baseline 截止 2020-09-23。")
    lines.append("- 原始特征只有 `open/high/low/close/volume/factor/change`，**没有 `$vwap` 和 `$amount`**；Alpha158 中依赖 VWAP 的列会偏弱或为 NaN；")
    lines.append("- 没有 ST、停牌、每日涨跌停价数据；回测用 `limit_threshold=0.095` 近似；")
    lines.append("- 数据截止 2022-12-30，不能用于 2023 年后的结论；")
    lines.append("- 没有做 Tushare/Wind 交叉对账（当前无 Tushare token），数据质量结论仅限抽样检查。")
    lines.append("")
    lines.append("### 2.3 数据质量扫描（`scripts/data_quality.py`）")
    lines.append("")
    if quality:
        lines.append("| 检查项 | 结果 |")
        lines.append("|---|---|")
        lines.append(f"| 原始日历 | {quality.get('calendar_start')} 至 {quality.get('calendar_end')}（{quality.get('calendar_days')} 天） |")
        bench = quality.get("benchmark_extremes", {}).get("sh000300", {})
        for dt, ret in bench.items():
            lines.append(f"| 基准极端日收益 | {dt}: {pct(ret)} |")
        lines.append(f"| 抽样股票数 | {quality.get('sampled_stocks')} |")
        lines.append(f"| 2020-09-28 因子跳变股票数 | {quality.get('sampled_stocks_with_factor_jumps')} |")
        lines.append(f"| 建议可用截止日 | {quality.get('recommended_usable_end')} |")
        lines.append("")
        for note in quality.get("notes", []):
            if "2020-09-28" in note:
                lines.append("- 2020-09-28 是供应商价格尺度断裂：复权收盘价重置、成交量跳变；标签和回测必须截止到 2020-09-23。")
            elif "2006-05-12" in note:
                lines.append("- 2006-05-12 是更早的基准断裂，不在本次使用区间内。")
            else:
                lines.append(f"- {note}")
        lines.append("")
    lines.append("## 3. 基线设计")
    lines.append("")
    lines.append("| 项目 | 配置 |")
    lines.append("|---|---|")
    lines.append("| 特征 | Qlib Alpha158（158 个量价特征） |")
    lines.append("| 模型 | LightGBM 回归 |")
    lines.append("| 标签 | `Ref($close, -2) / Ref($close, -1) - 1`（下一交易日收益） |")
    lines.append(f"| 训练集 | {segments['train'][0]} 至 {segments['train'][1]} |")
    lines.append(f"| 验证集 | {segments['valid'][0]} 至 {segments['valid'][1]} |")
    lines.append(f"| 测试集 | {segments['test'][0]} 至 {segments['test'][1]} |")
    lines.append("| 组合 | TopkDropout，topk=50，n_drop=5，hold_thresh=1 |")
    lines.append("| 成交 | 信号滞后 1 个交易日，`deal_price=close` |")
    lines.append("| 成本 | 买入 5bp，卖出 15bp，最低 5 元，100 股一手 |")
    lines.append("| 基准 | SH000300 |")
    lines.append("")
    lines.append("模型参数：")
    lines.append("")
    lines.append("| 参数 | 值 |")
    lines.append("|---|---|")
    for k in [
        "learning_rate",
        "colsample_bytree",
        "subsample",
        "subsample_freq",
        "lambda_l1",
        "lambda_l2",
        "max_depth",
        "num_leaves",
        "num_threads",
        "num_boost_round",
        "early_stopping_rounds",
    ]:
        lines.append(f"| {k} | {model.get(k)} |")
    lines.append("")
    lines.append("## 4. 结果")
    lines.append("")
    lines.append("### 4.1 因子级指标（测试集）")
    lines.append("")
    lines.append("| 指标 | RankIC | Pearson IC |")
    lines.append("|---|---:|---:|")
    lines.append(f"| 均值 | {num(rank.get('mean'), 4)} | {num(pearson.get('mean'), 4)} |")
    lines.append(f"| 标准差 | {num(rank.get('std'), 4)} | {num(pearson.get('std'), 4)} |")
    lines.append(f"| ICIR（日） | {num(rank.get('icir'), 4)} | {num(pearson.get('icir'), 4)} |")
    lines.append(f"| ICIR（年化） | {num(rank.get('icir_annualized'), 4)} | {num(pearson.get('icir_annualized'), 4)} |")
    lines.append(f"| t 统计量 | {num(rank.get('t_stat'), 3)} | {num(pearson.get('t_stat'), 3)} |")
    lines.append(f"| Newey-West t | {num(rank.get('newey_west_t'), 3)} | {num(pearson.get('newey_west_t'), 3)} |")
    lines.append(f"| 正 IC 占比 | {pct(rank.get('positive_ratio'))} | {pct(pearson.get('positive_ratio'))} |")
    lines.append(f"| 有效交易日 | {rank.get('n_dates')} | {pearson.get('n_dates')} |")
    lines.append("")
    lines.append("### 4.2 分层收益（下一日，按因子值 5 分组）")
    lines.append("")
    lines.append("| 分位 | 平均下一日收益 |")
    lines.append("|---|---:|")
    for k, v in quant.get("mean_forward_return", {}).items():
        lines.append(f"| Q{k} | {pct(v, 3)} |")
    lines.append("")
    lines.append(f"- 多空组合（Q4−Q0）日均：{pct(quant.get('long_short_daily_mean'), 3)}")
    lines.append(f"- 多空组合年化均值：{pct(quant.get('long_short_annualized_mean'), 2)}")
    lines.append(f"- 分层单调性（Spearman）：{num(quant.get('monotonicity_spearman'), 3)}")
    lines.append(f"- Top 20% 组合日均换手：{pct(metrics.get('top_quintile_turnover'), 2)}")
    lines.append("")
    lines.append("### 4.3 组合回测（测试集，含成本）")
    lines.append("")
    lines.append("| 指标 | 组合 | 超额（不含成本） | 超额（含成本） |")
    lines.append("|---|---:|---:|---:|")
    lines.append(f"| 年化收益 | {pct(portfolio.get('annualized_return'))} | {pct(excess_no_cost.get('annualized_return'))} | {pct(excess_cost.get('annualized_return'))} |")
    lines.append(f"| 日收益均值 | {pct(portfolio.get('mean'), 3)} | {pct(excess_no_cost.get('mean'), 3)} | {pct(excess_cost.get('mean'), 3)} |")
    lines.append(f"| 日收益波动 | {pct(portfolio.get('std'), 3)} | {pct(excess_no_cost.get('std'), 3)} | {pct(excess_cost.get('std'), 3)} |")
    lines.append(f"| 信息比率 | {num(portfolio.get('information_ratio'), 3)} | {num(excess_no_cost.get('information_ratio'), 3)} | {num(excess_cost.get('information_ratio'), 3)} |")
    lines.append(f"| 最大回撤 | {pct(portfolio.get('max_drawdown'))} | {pct(excess_no_cost.get('max_drawdown'))} | {pct(excess_cost.get('max_drawdown'))} |")
    lines.append("")
    lines.append("### 4.4 图表")
    lines.append("")
    lines.append("![Cumulative RankIC](assets/cumulative_rank_ic.png)")
    lines.append("")
    lines.append("![Quantile returns](assets/quantile_returns.png)")
    lines.append("")
    lines.append("![Cumulative return](assets/cumulative_return.png)")
    lines.append("")
    lines.append("![Drawdown](assets/drawdown.png)")
    lines.append("")
    lines.append("## 5. 如何解读")
    lines.append("")
    lines.append("- **因子级**：看 RankIC 均值、ICIR、t 统计量和分层单调性。")
    lines.append("- **组合级**：看含成本后的超额、信息比率、最大回撤和换手。")
    lines.append("- **两者不一致是正常的**：因子有弱 IC，不代表 TopkDropout 组合一定跑赢基准；")
    lines.append("  组合还受股票池、权重、换手、成本和基准影响。")
    lines.append("- **不要把这次高年化当成策略结论**：测试期只有 419 个交易日，且 2019–2020 是单一有利行情；股票池还带 2020 年成分前视。Stage 2 会用 walk-forward、FDR/DSR/PBO 和锁定 OOS 重新评估。")
    lines.append("- 本次结果应作为**基线**，不是投资结论。")
    lines.append("")
    lines.append("## 6. 已知限制（必须在面试中主动说）")
    lines.append("")
    lines.append("1. **数据不是 PIT**：Qlib `cn_data_v3` 是 Yahoo 派生数据，缺少基本面公告日、ST、停复牌、每日涨跌停价。")
    lines.append("2. **2020-09-28 价格尺度断裂**：复权价重置、成交量跳变，2020-09-28 之后的样本被排除；因此测试期只到 2020-09-23。")
    lines.append("3. **股票池有 2020 年成分前视**：`csi300_frozen` 用 2020-09-25 成分构造，不是 point-in-time。")
    lines.append("4. **没有 VWAP/成交额**：Alpha158 中部分特征退化，基线偏弱。")
    lines.append("5. **执行假设简化**：信号滞后 1 日、收盘成交、固定 9.5% 涨跌停阈值，没有盘口/冲击成本/容量约束。")
    lines.append("6. **只做了单次时间切分**：还不是 walk-forward，也没有多重检验校正。")
    lines.append("7. **Windows/Qlib 工程限制**：Qlib 多进程在中文路径下会失败，项目必须放在 ASCII 路径；脚本用 `C.joblib_backend` 与新 MLflow 的 `MLFLOW_ALLOW_FILE_STORE` 做了兼容。")
    lines.append("")
    lines.append("## 7. 复现方式")
    lines.append("")
    lines.append("```bash")
    lines.append("cd D:/quant/llm-gp-factor-lab")
    lines.append("# 1) 环境（首次）")
    lines.append("# py -3.12 -m venv .venv")
    lines.append("# .venv/Scripts/python.exe -m pip install -r requirements.txt")
    lines.append("")
    lines.append("# 2) 数据覆盖检查")
    lines.append(".venv/Scripts/python.exe scripts/check_data.py")
    lines.append("")
    lines.append("# 3) 快速 smoke（约 2 分钟）")
    lines.append(".venv/Scripts/python.exe scripts/run_baseline.py --quick")
    lines.append("")
    lines.append("# 4) 完整 baseline")
    lines.append(".venv/Scripts/python.exe scripts/run_baseline.py")
    lines.append("")
    lines.append("# 5) 重新生成本报告")
    lines.append(".venv/Scripts/python.exe scripts/build_report.py")
    lines.append("```")
    lines.append("")
    lines.append("产物目录：`artifacts/baseline/`（预测、IC、净值、指标 JSON、图表）。")
    lines.append("")
    lines.append("## 8. 下一阶段")
    lines.append("")
    lines.append("- **Stage 2**：数据质量补强（PIT、ST、涨跌停、停复牌）；")
    lines.append("- **Stage 3**：表达式 Validator + Evaluator；")
    lines.append("- **Stage 4**：S1 单因子评估标准化（IC/ICIR/分层/衰减/换手）；")
    lines.append("- **Stage 5**：LLM 生成候选；")
    lines.append("- **Stage 6**：GP 进化；")
    lines.append("- **Stage 7**：去重与四层共线性；")
    lines.append("- **Stage 8–10**：S2 稳健性、S3 组合、消融与锁定 OOS。")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("报告结束。数字来自 `artifacts/baseline/metrics.json`，图表来自同目录 PNG。")
    lines.append("")

    out = REPORT_DIR / "Stage0_Stage1_项目报告.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}")
    print(f"assets -> {ASSETS_DIR}")


if __name__ == "__main__":
    main()
