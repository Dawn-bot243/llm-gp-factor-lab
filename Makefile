PY := .venv/Scripts/python.exe

.PHONY: setup universe data data-quality data-quality-quick mask snapshot investment-data pit-universe dolt-limits pit-constraints tushare-daily patch-qlib patch-qlib-apply patch-qlib-qfq baseline-quick baseline baseline-masked baseline-compare baseline-patch-compare baseline-pit baseline-pit-compare validate-alpha158 factor-eval-smoke factor-eval-ma5-full verify-factor stage3-check test report

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

test:
	$(PY) -m pytest

report:
	$(PY) scripts/build_report.py
