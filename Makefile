PY := .venv/Scripts/python.exe
LLM_MODEL ?= deepseek-chat

.PHONY: setup universe data data-quality data-quality-quick mask snapshot investment-data pit-universe dolt-limits pit-constraints tushare-daily patch-qlib patch-qlib-apply patch-qlib-qfq baseline-quick baseline baseline-masked baseline-compare baseline-patch-compare baseline-pit baseline-pit-compare validate-alpha158 factor-eval-smoke factor-eval-ma5-full verify-factor stage3-check s1-calibration s1-alpha158 stage4-check llm-mock llm-generate llm-screen stage5-check gp-smoke gp-evolve stage6-check dedupe stage7-check s2 stage8-check sector-map portfolio portfolio-quick stage9-plot stage9-check stage10-universe stage10-constraints stage10-random stage10-gp-only stage10-ablation stage10-plot stage10-check stage10b-random stage10b-gp-only stage10b-ablation stage10b-plot stage10b-check stage10c-fetch stage10c-constraints stage10c-validate stage10c-ablation stage10c-plot stage10c-bootstrap stage10c-report stage10c-check stage11-demo stage11-check test report

setup:
	$(PY) -m pip install -r requirements.txt

universe:
	$(PY) scripts/build_universe.py

data:
	$(PY) scripts/check_data.py

data-quality:
	$(PY) scripts/data_quality.py

data-quality-quick:
	$(PY) scripts/data_quality.py --skip-external

mask:
	$(PY) scripts/build_quality_mask.py

snapshot:
	$(PY) scripts/build_pit_snapshot.py

investment-data:
	$(PY) -I scripts/download_investment_data_parallel.py --workers 8 --chunk-mb 32 --extract

pit-universe:
	$(PY) -I scripts/build_pit_universe.py

dolt-limits:
	$(PY) -I scripts/fetch_dolt_limits.py --workers 4

pit-constraints:
	$(PY) -I scripts/build_pit_constraints.py

tushare-daily:
	$(PY) scripts/fetch_tushare_daily.py
	$(PY) scripts/validate_tushare_daily.py

patch-qlib:
	$(PY) scripts/patch_qlib_holes.py

patch-qlib-apply:
	$(PY) scripts/patch_qlib_holes.py --apply
	$(PY) scripts/patch_qlib_qfq.py --include-placeholders --apply

patch-qlib-qfq:
	$(PY) scripts/patch_qlib_qfq.py --include-placeholders

baseline-quick:
	$(PY) scripts/run_baseline.py --quick --output-dir artifacts/baseline_quick_pit

baseline:
	$(PY) scripts/run_baseline.py --output-dir artifacts/baseline_pit

baseline-masked:
	$(PY) scripts/run_baseline.py --mask-file data/processed/data_quality_mask.json --output-dir artifacts/baseline_masked

baseline-compare:
	$(PY) scripts/compare_baseline_runs.py

baseline-patch-compare:
	$(PY) scripts/compare_baseline_patch.py

baseline-pit:
	$(PY) scripts/run_baseline.py --output-dir artifacts/baseline_pit

baseline-pit-compare:
	$(PY) scripts/compare_baseline_pit.py

validate-alpha158:
	$(PY) scripts/validate_expression.py --alpha158

factor-eval-smoke:
	$(PY) scripts/evaluate_factor.py --expr 'Mean($close, 5)/$close - 1' --name ma5 \
	  --start 2020-01-02 --end 2020-09-23 \
	  --output-dir artifacts/factor_eval/ma5_smoke --horizons 1,5

factor-eval-ma5-full:
	$(PY) scripts/evaluate_factor.py --expr 'Mean($close, 5)/$close - 1' --name ma5_full \
	  --start 2010-01-01 --end 2020-09-23 \
	  --output-dir artifacts/factor_eval/ma5_full --horizons 1,2,5,10,20

verify-factor: factor-eval-ma5-full
	$(PY) -I scripts/verify_factor_eval.py

stage3-check: validate-alpha158 verify-factor test

s1-calibration:
	$(PY) scripts/screen_factors.py --known --alpha158-sample 5 \
	  --output-dir artifacts/s1/calibration --store-dir data/factor_store

s1-alpha158:
	$(PY) scripts/screen_factors.py --alpha158-sample 10 \
	  --output-dir artifacts/s1/alpha158 --store-dir data/factor_store

stage4-check: s1-calibration test

llm-mock:
	$(PY) scripts/generate_factors.py --mock --n 50 --no-screen \
	  --run-dir artifacts/mining/mock

llm-generate:
	$(PY) scripts/generate_factors.py --model $(LLM_MODEL) --n 200 --no-screen --seed-alpha158 \
	  --run-dir data/mining/run_stage5_200

llm-screen:
	$(PY) scripts/generate_factors.py --screen-only --run-dir data/mining/run_stage5_200

stage5-check: llm-mock test

gp-smoke:
	$(PY) scripts/run_gp.py --budget 50 --population 20 --generations 3 \
	  --output-dir artifacts/mining/gp_smoke

gp-evolve:
	$(PY) scripts/run_gp.py --budget 2000 \
	  --output-dir data/mining/gp_run_full_v2

stage6-check: gp-smoke test

dedupe:
	$(PY) scripts/dedupe_factors.py

stage7-check: dedupe test

s2:
	$(PY) scripts/s2_robustness.py

stage8-check: s2 test

sector-map:
	$(PY) -I scripts/fetch_sector_map.py

portfolio:
	$(PY) scripts/run_portfolio.py

portfolio-quick:
	$(PY) scripts/run_portfolio.py --quick --no-reconcile \
	  --output-dir artifacts/stage9_quick

stage9-plot:
	$(PY) -I scripts/plot_stage9.py

stage9-check: stage9-plot test

stage10-universe:
	$(PY) -I scripts/build_stage10_universe.py --oos-end 2023-06-12

stage10-constraints:
	$(PY) -I scripts/build_stage10_constraints.py

stage10-random:
	$(PY) -I scripts/run_random_search.py --budget 500 --seed 123

stage10-gp-only:
	$(PY) -I scripts/run_gp_worker.py --budget 500 --seed 43 \
	  --output-dir data/mining/stage10_gp_only

stage10-ablation:
	$(PY) -I scripts/run_ablation.py

stage10-plot:
	$(PY) -I scripts/plot_stage10.py

stage10-check: stage10-plot test

stage10b-random:
	for s in 101 102 103; do \
	  $(PY) -I scripts/run_random_search.py --budget 230 --seed $$s \
	    --output-dir data/mining/stage10b_random_seed$$s; \
	done

stage10b-gp-only:
	for s in 201 202 203; do \
	  $(PY) -I scripts/run_gp_worker.py --budget 230 --seed $$s \
	    --output-dir data/mining/stage10b_gp_seed$$s; \
	done

stage10b-ablation:
	$(PY) -I scripts/run_ablation.py --config configs/stage10b.yaml \
	  --output-dir artifacts/stage10b

stage10b-plot:
	$(PY) -I scripts/plot_stage10b.py

stage10b-check: stage10b-plot test

stage10c-fetch:
	$(PY) -I scripts/fetch_zzshare_daily.py

stage10c-constraints: stage10c-fetch
	$(PY) -I scripts/build_stage10c_constraints.py

stage10c-validate:
	$(PY) -I scripts/validate_stage10c_constraints.py

stage10c-ablation:
	$(PY) -I scripts/run_ablation.py --config configs/stage10c.yaml \
	  --output-dir artifacts/stage10c

stage10c-plot:
	$(PY) -I scripts/plot_stage10c.py

stage10c-bootstrap:
	$(PY) -I scripts/bootstrap_stage10c.py

stage10c-report:
	$(PY) -I scripts/report_stage10c.py

stage10c-check: stage10c-plot stage10c-bootstrap test

stage11-demo:
	$(PY) -m jupyter nbconvert --to notebook --execute --inplace \
	  --ExecutePreprocessor.kernel_name=llm-gp-factor-lab \
	  --ExecutePreprocessor.timeout=180 demo/stage11_demo.ipynb

stage11-check:
	$(PY) -m ruff check . --exclude .venv
	$(PY) -m mypy src scripts
	$(PY) -m pytest -q

test:
	$(PY) -m pytest

report:
	$(PY) scripts/build_report.py
