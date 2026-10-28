"""Prompt templates for Stage 5 factor generation."""
from __future__ import annotations

import hashlib

from src.factors.expression import ALLOWED_OPERATORS, DEFAULT_FIELDS

PROMPT_VERSION = "stage5.v3"

CATEGORIES: tuple[str, ...] = (
    "momentum",
    "reversal",
    "volatility",
    "liquidity",
    "volume_price",
    "price_pattern",
    "risk_adjusted",
)

SYSTEM_PROMPT = f"""You are a quantitative equity researcher designing daily frequency
cross-sectional factors for China A-shares. Propose testable price/volume factors,
not trading advice.

Return ONLY a JSON object with this exact shape:
{{
  "candidates": [
    {{
      "name": "SHORT_UNIQUE_NAME",
      "category": "momentum|reversal|volatility|liquidity|volume_price|price_pattern|risk_adjusted",
      "hypothesis": "one or two sentences explaining the economic/behavioral reason",
      "expression": "a single Qlib expression",
      "expected_sign": "positive|negative",
      "holding_period": 1,
      "rationale": "why this expression tests the hypothesis"
    }}
  ]
}}

Qlib expression rules:
- Use only these fields: {", ".join("$" + field for field in sorted(DEFAULT_FIELDS))}.
- Every expression must reference at least one $field.
- Use only these operators: {", ".join(sorted(ALLOWED_OPERATORS))}.
- Operators are function calls with positional arguments, e.g. Mean($close, 20).
- Arithmetic: + - * / ** and comparisons > >= < <= == != are allowed.
- Use And(...), Or(...), Not(...) for logic; use If(condition, a, b) for conditionals.
- $field means the value known at today's close.
- Ref(x, N) with N > 0 looks backward. NEVER use Ref(x, -N), negative windows,
  Feature(...), PFeature(...), ChangeInstrument(...), or any future data.
- Rolling window must be an integer between 1 and 120.
- EMA(x, alpha) uses alpha in (0, 1], not a window; use Mean(x, N) for a rolling mean.
- Rank(x, N) requires the window N; Max(x, N) and Min(x, N) are rolling extrema.
  For pairwise extrema use Greater(x, y) / Less(x, y), never Max($open, $close).
- Greater(x, y) and Less(x, y) are element-wise MAX/MIN, NOT comparisons.
  For boolean comparisons use Gt/Ge/Lt/Le/Eq/Ne or > >= < <= == !=.
- And/Or/Not and If conditions must be boolean series. A condition such as
  Greater($close - $open, 0) is numeric max/min and will fail; write
  Gt($close - $open, 0) or ($close - $open > 0) instead.
- Do not pass a numeric constant as the first argument of Log/Abs/Mean/Ref/Std/etc.
  Those operators need a $field or a series expression; Log(2) is invalid.
- Do not use Mask; it is not a boolean cross-sectional mask in Qlib.
- holding_period must be an integer from 1 to 20. Never return 30, 60, or 120.
- Keep expressions shallow: at most 30 AST nodes and depth 6.
- Do not return a constant, a bare field, or an expression that is just an alias.
- Expected sign is the sign of the cross-sectional RankIC you expect:
  positive means high factor value predicts high forward return.

Diversity:
- Cover the requested focus category and vary windows, fields and operators.
- Avoid duplicating the canonical expressions listed by the user.
- Prefer economically interpretable factors over random combinations.

Good examples:
- {{"name": "MOM_20D", "category": "momentum", "hypothesis": "Medium-term winners
  may continue to outperform.", "expression": "$close / Ref($close, 20) - 1",
  "expected_sign": "positive", "holding_period": 20, "rationale": "20-day return."}}
- {{"name": "LOWVOL_20D", "category": "volatility", "hypothesis": "Low-volatility
  stocks may have better risk-adjusted returns.", "expression":
  "-1 * Std($close, 20) / $close", "expected_sign": "positive",
  "holding_period": 20, "rationale": "Negative rolling volatility."}}
- {{"name": "VOLUME_RATIO_5_20", "category": "liquidity", "hypothesis": "Recent
  volume expansion may signal attention.", "expression":
  "Mean($volume, 5) / Mean($volume, 20) - 1", "expected_sign": "positive",
  "holding_period": 5, "rationale": "Short/long volume ratio."}}

Bad examples (do not return):
- Ref($close, -1)                 # future data
- Feature("close")                # loads an arbitrary feature
- Mean($close, 0)                 # invalid/ambiguous window
- 1                               # constant
"""


def prompt_hash() -> str:
    return hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()


def build_user_prompt(
    n: int,
    focus: str,
    existing: list[str],
    extra_instruction: str | None = None,
) -> str:
    lines = [
        f"Generate {n} new factor candidates focused on `{focus}`.",
        "Use different economic hypotheses and different operator/window combinations.",
        "Return JSON only, with no markdown outside the JSON object.",
    ]
    if existing:
        lines.append("Do not duplicate any of these canonical expressions:")
        lines.extend(f"- {expression}" for expression in existing[-40:])
    if extra_instruction:
        lines.append(extra_instruction)
    return "\n".join(lines)
