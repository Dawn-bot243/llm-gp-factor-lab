"""Stage 9: S3 portfolio construction, costs, constraints and attribution."""

from src.portfolio.attribution import brinson_attribution, style_exposures
from src.portfolio.combine import (
    FactorSpec,
    ScoreResult,
    build_combination,
    load_factor_panels,
    open_to_open_returns,
)
from src.portfolio.costs import CostConfig
from src.portfolio.metrics import performance_metrics
from src.portfolio.simulator import (
    PortfolioConfig,
    PortfolioResult,
    TradingConstraints,
    run_portfolio,
)

__all__ = [
    "CostConfig",
    "FactorSpec",
    "PortfolioConfig",
    "PortfolioResult",
    "ScoreResult",
    "TradingConstraints",
    "brinson_attribution",
    "build_combination",
    "load_factor_panels",
    "open_to_open_returns",
    "performance_metrics",
    "run_portfolio",
    "style_exposures",
]
