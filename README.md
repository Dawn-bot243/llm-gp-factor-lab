# LLM-GP Factor Lab（LLM + GP 因子挖掘实验室）

一个 LLM + GP 因子挖掘的研究原型，带多阶段验证流程。
项目分阶段推进：先建立可复现的 Qlib 基线，再做因子生成，最后进行严格筛选。

> 所有项目报告统一放在 `项目报告/`（已加入 `.gitignore`，只在本地保留）；
> GitHub 仓库只上传代码、配置、脚本和测试，不含 `项目报告/`、`data/`、
> `artifacts/`、`.env`；README 中指向这些本地目录的链接在 GitHub 上不可用。
> `reports/` 只保留运行日志。

## 进展状态

- **阶段 0**：环境、Qlib 数据、数据覆盖报告 —— 已完成。
- **阶段 1**：`Alpha158 + LightGBM + TopkDropout` 基线 —— 已完成；历史产物在
  `artifacts/baseline/`。
- **阶段 2**：数据质量检查、空洞修补、外部对账、数据快照 —— 已完成；
  旧数据报告见 [`项目报告/stage2_completion_cn_data_v3.md`](项目报告/stage2_completion_cn_data_v3.md)。
- **阶段 2.7**：数据源切换（`investment_data`）+ PIT CSI300 + PIT baseline ——
  已完成；见 [`项目报告/stage2_completion.md`](项目报告/stage2_completion.md)。
- **阶段 3**：表达式 Validator + Evaluator、PIT 因子评估、批量榜单 —— 已完成；
  见 [`项目报告/Stage3_表达式校验与评估_项目报告.md`](项目报告/Stage3_表达式校验与评估_项目报告.md)。
- **阶段 4+**：S1 单因子标准化、LLM/GP 挖掘、多阶段筛选 —— 尚未开始。

## 数据源与环境

- Windows 11 原生环境，Python 3.12
- Qlib 0.9.7，LightGBM 4.7
- **主数据源**：`chenditc/investment_data`（Qlib bin，Apache-2.0 仓库）
  - 解压目录：`D:/qlib_data/cn_data_investment_data/extracted/qlib_bin`
  - Release：`2026-10-25`（target trade date 2026-10-23）
  - 压缩包 SHA256：`0a2974da97b4737e503ecddeea8bb3b2179e0e16ea0fb19fa373ff14155bf315`
  - 多源合并（Wind/Caihui/Tushare/AkShare/Baostock/Yahoo），不是交易所官方数据
  - 日历：2000-01-04 至 2026-10-23，6,494 天；本次使用 **2010-01-01 至 2020-09-23**
  - 字段：`open/high/low/close/volume/factor/adjclose/amount/vwap/change`
  - `volume` 是调整后的**手**（1 手 = 100 股）：
    `raw_volume_shares = volume * factor * 100`，`turnover_cny = volume * close * 100`
- **旧 fallback**：`D:/qlib_data/cn_data_v3`（Yahoo 社区数据，已补洞；快照
  `qlib-2f1550fea963`）
- 外部数据：
  - Tushare `daily`：300 只标的的行情缓存与对账
  - AkShare Sina：原始/前复权价，用于对账
  - DoltHub `final_a_stock_limit`：PIT 涨跌停价（已导出，2,609 天 / 782,425 行）
- 数据快照：`qlib-pit-7c6be6609542`（[`项目报告/qlib_snapshot.json`](项目报告/qlib_snapshot.json)）

## 股票池（PIT CSI300）

主口径使用 `instruments/csi300.txt` 的 **PIT 区间**，不是 frozen 成分：

- 2010–2020-09-23：661 只历史成分，782,700 个成员-日；
- 每天恰好 **300** 只（min = max = median = 300）；
- baseline 使用 `PITAlpha158`：先加载 `csi300` 的 union，再按日度成员表过滤。

旧的 `csi300_frozen`（2020 成分回溯）只保留为历史 fallback，不再作为主结果。
详见 [`项目报告/stage2_completion.md`](项目报告/stage2_completion.md) 和
[`项目报告/investment_data_validation.md`](项目报告/investment_data_validation.md)。

## 数据质量结论

完整新数据验证见 [`项目报告/investment_data_validation.md`](项目报告/investment_data_validation.md)。

- **日历**：2010–2020-09-23 的 2,609 个官方交易日 **0 缺失**；旧数据缺 47 天的问题已消失。
- **2020-09-28**：新数据没有复权因子重置；SH600000 原始价 9.47→9.46 连续，
  SH000300 原始点位 4570→4582 连续。
- **数据空洞**：相对 AkShare 的 `data_hole` = **0**；旧数据补洞后仍有 7 个标的-日。
- **NaN**：close NaN 21,551（3.30%），主要是停牌/无交易。
- **PIT**：PIT CSI300 每天恰好 300 只，成分前视已修复。
- **字段**：新增 `vwap`、`amount`、`change`、`adjclose`；Alpha158 的 VWAP 特征不再缺失。
- **价格异常**：raw 硬跳变 120 个，全部是公司行为（adjusted return 都在 ±21% 内）；
  拆股/公司行为日 283 个；2020-09-28 因子重置 0。
- **对账**：新数据 vs AkShare 原始收益 **99.999%**、复权收益 **99.38%**；
  成交额修正 volume 单位后 **100%**；旧/新 raw return 匹配 99.79%。
- **ST/涨跌停**：新 Qlib 归档不含 ST/limit；已从 DoltHub `final_a_stock_limit`
  导出 2,609 天、782,425 行，并生成 PIT 约束面板：停牌 24,564 个标的-日、
  ST 640 个标的-日、涨/跌停触及 9,594 / 7,288。20% 板块的 ST 仍标 unknown。

旧数据的 47 个缺日、2020-09-28 复权重置、2,561 个空洞、7 个残余空洞等诊断，
保留在 [`项目报告/data_quality_pre_patch.md`](项目报告/data_quality_pre_patch.md) 和
[`项目报告/stage2_completion_cn_data_v3.md`](项目报告/stage2_completion_cn_data_v3.md)。

## 快速开始

```bash
cd D:/quant/llm-gp-factor-lab

# 1) 环境（首次执行）
# py -3.12 -m venv .venv
# .venv/Scripts/python.exe -m pip install -r requirements.txt

# 2) 下载并校验 investment_data（GitHub API asset 端点）
.venv/Scripts/python.exe -I scripts/download_investment_data_parallel.py \
  --workers 8 --chunk-mb 32 --extract

# 3) 构建 PIT CSI300 成员表
.venv/Scripts/python.exe -I scripts/build_pit_universe.py

# 4) 数据质量：新数据探针 + 新旧对比
.venv/Scripts/python.exe -I scripts/probe_investment_data.py \
  --provider-uri D:/qlib_data/cn_data_investment_data/extracted/qlib_bin
.venv/Scripts/python.exe -I scripts/compare_providers.py \
  --new D:/qlib_data/cn_data_investment_data/extracted/qlib_bin \
  --old D:/qlib_data/cn_data_v3

# 5) 完整质量检查（300 只旧 frozen 子集，结果写到临时目录）
.venv/Scripts/python.exe -I scripts/data_quality.py \
  --provider-uri D:/qlib_data/cn_data_investment_data/extracted/qlib_bin \
  --universe data/universes/csi300_frozen.txt \
  --reports-dir 项目报告/investment_data_check \
  --processed-dir data/processed/investment_data_check \
  --start 2010-01-01 --end 2020-09-23 --sample-size 100

# 6) 生成 PIT 快照
.venv/Scripts/python.exe -I scripts/build_pit_snapshot.py

# 7) PIT baseline
.venv/Scripts/python.exe scripts/run_baseline.py --output-dir artifacts/baseline_pit

# 8) 对比与校验
.venv/Scripts/python.exe scripts/compare_baseline_pit.py
.venv/Scripts/python.exe scripts/validate_baseline.py --artifacts artifacts/baseline_pit

# 9) Dolt 涨跌停导出 + PIT 约束面板（S3 前）
.venv/Scripts/python.exe -I scripts/fetch_dolt_limits.py --workers 4 --chunk-days 3
.venv/Scripts/python.exe -I scripts/build_pit_constraints.py

# 10) Stage 3：表达式校验 + 单因子评估（PIT）
.venv/Scripts/python.exe scripts/validate_expression.py --alpha158
.venv/Scripts/python.exe scripts/evaluate_factor.py \
  --expr 'Mean($close, 5)/$close - 1' --name ma5 \
  --output-dir artifacts/factor_eval/ma5_full
.venv/Scripts/python.exe scripts/evaluate_factor.py --alpha158-sample 3
.venv/Scripts/python.exe -I scripts/verify_factor_eval.py
```

旧 fallback（`cn_data_v3`）的补洞/baseline 命令保留在
[`项目报告/stage2_completion_cn_data_v3.md`](项目报告/stage2_completion_cn_data_v3.md)。

## 产物

PIT 主口径：

- `artifacts/baseline_pit/`：PIT baseline 的 metrics、预测、IC、组合报告、图表
- `data/processed/pit_csi300_membership.parquet`：PIT 日度成员
- `data/processed/pit_csi300_constraints.parquet`：PIT ST/涨跌停/停牌面板
- `项目报告/pit_constraints_summary.json`：PIT 约束摘要
- `data/manifests/qlib_snapshot.json`：快照 `qlib-pit-7c6be6609542`
- `项目报告/investment_data_validation.md`：新数据验证报告
- `项目报告/provider_comparison.json`：新旧数据对比
- `项目报告/investment_data_check/`：新数据质量检查
- `项目报告/baseline_pit_comparison.md` / `.json`：旧 frozen vs 新 PIT
- `项目报告/stage2_completion.md`：Stage 2 完成报告

Stage 3：

- `artifacts/factor_eval/ma5_full/`：MA5 全区间评估（metrics/IC/分层/多空）
- `artifacts/factor_eval/alpha158_sample3/`：3 条 Alpha158 因子的批量榜单
- `data/processed/factor_cache/`：按表达式/config 哈希缓存的因子与标签面板
- `scripts/validate_expression.py`、`scripts/evaluate_factor.py`：Stage 3 CLI
- `scripts/verify_factor_eval.py`：MA5 的独立 pandas 复核
- `项目报告/Stage3_表达式校验与评估_项目报告.md`：Stage 3 报告

旧数据/历史：

- `artifacts/baseline/`、`baseline_masked/`、`baseline_patched/`
- `项目报告/data_quality.md`、`data_quality_pre_patch.md`
- `项目报告/qlib_snapshot_cn_data_v3.json`
- `data/raw/{baostock,akshare,tushare,dolt}/`、`data/raw/manifests/`

## 基线设计

| 项目 | 取值 |
|---|---|
| 数据源 | `investment_data` Qlib bin |
| 股票池 | PIT CSI300（每天 300 只） |
| 特征 | Qlib Alpha158 |
| 模型 | LightGBM 回归 |
| 标签 | 下一个交易日收益 |
| 数据划分 | 训练 2010–2016，验证 2017–2018，测试 2019–2020-09-23 |
| 组合 | TopkDropout，topk=50，n_drop=5 |
| 信号时点 | 信号平移 1 个交易日，`deal_price=close` |
| 成本 | 买入 5bp，卖出 15bp，最低 5 元，100 股一手 |
| 基准 | SH000300 |

### PIT baseline 结果（单次测试切分，并非经过验证的策略）

| 指标 | 数值 |
|---|---:|
| 测试 IC 天数 | 422 |
| RankIC 均值 | 0.0401 |
| ICIR（日频） | 0.325 |
| Newey-West t 值 | 5.97 |
| 多空年化收益（毛收益） | 44.5% |
| Top-20% 换手率（日频） | 55.3% |
| 含成本超额收益（年化） | 14.4% |
| 含成本超额信息比率 | 1.70 |
| 组合最大回撤 | -17.9% |

> 旧 frozen 数据为 23.5%/IR 2.40；新 PIT 降到 14.4%/1.70。这不是退步：
> 旧 frozen 用 2020 年成分回看 2010 年，会高估收益。新 PIT 更可信。
> 完整对比见 [`项目报告/baseline_pit_comparison.md`](项目报告/baseline_pit_comparison.md)。
> 这是基线参考，不是投资结论；测试窗口是单一有利行情，尚无 walk-forward/OOS。

## Stage 3 表达式校验与评估

- **Validator**：用 `ast` 解析表达式，不允许 Python `eval` 直接接触原始字符串；
  字段/算子白名单；拒绝 `Ref($field, -k)`、负窗口、`Feature(...)`、
  `ChangeInstrument(...)`、过复杂表达式；输出 canonical 表达式和 SHA256。
- **Evaluator**：固定快照 `qlib-pit-7c6be6609542` + PIT CSI300；
  标签与 baseline 一致（`Ref($close,-2)/Ref($close,-1)-1`）；计算
  RankIC/ICIR/Newey-West t、分层收益、IC 衰减、换手和 rank 自相关；
  因子/标签面板按 config 哈希缓存。
- **自检**：Alpha158 的 158 条表达式全部通过 Validator，45 个算子通过 Qlib
  解析检查；MA5 全区间 RankIC 0.0414、ICIR 0.224、NW t 11.7、
  覆盖率 96.9%；独立 pandas 复核的 2609 个共同交易日最大 RankIC 差
  `0.00027`。
- 3 条 Alpha158 样本（MIN60/CNTN10/KMID）的 RankIC 分别为
  0.0224/0.0202/-0.0299，仅用于验证评估链路，不是策略结论。

> Stage 3 只做“能安全算、能统一比较”，不做因子生成和组合回测；
> 全区间指标包含训练/验证段，不能当作样本外表现。

## 已知局限

- 新数据是多源合并数据，不是交易所官方；上游数据源条款需遵守。
- ST / 涨跌停 / 停复牌：Qlib 归档不含这些字段；已用 Dolt limit 生成 PIT 约束面板，
  但 20% 板块的 ST 状态仍为 unknown。
- 研究区间仍截止 2020-09-23；新数据可覆盖到 2026，但本次未扩展。
- 无分钟频/日内执行数据；`adjclose` 等额外字段未全部纳入特征。
- S3 组合回测前需要补齐 ST/limit/停复牌，并重新评估容量/成本。
- Stage 3 的指标是全区间单因子统计，没有 walk-forward、多重检验校正或
  锁定 OOS；这些留给 Stage 4+。
- Qlib 的 rolling 算子使用 `min_periods=1`，比较运算遇到 NaN 时可能产生
  0/False；低覆盖或高覆盖都不等于“数据无问题”，需要结合因子定义判断。

## Windows / Qlib 注意事项

- 项目路径保持 ASCII；原来的中文路径导致 Qlib 多进程 `spawn` 报
  `OSError: [Errno 22]`。
- 基线脚本在 `qlib.init()` 之后设置 `C.joblib_backend` 和 `C.kernels`，
  因为 `qlib.init()` 会重置配置。
- 较新的 MLflow 版本会拒绝默认的 file store；脚本在导入 Qlib 之前设置
  `MLFLOW_ALLOW_FILE_STORE=true`。
- GitHub Release 直链在本机不可达；下载脚本使用 GitHub API 的 release asset
  端点，并用 manifest 的 sha256 校验。
