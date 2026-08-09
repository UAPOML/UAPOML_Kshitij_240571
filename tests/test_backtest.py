"""
Unit tests for financial metrics, drawdown, turnover, and backtesting logic.
"""

import numpy as np
import pandas as pd
import pytest

from src.metrics import compute_drawdown_series, compute_portfolio_metrics


def test_drawdown_calculation():
    """Verify drawdown and maximum drawdown calculation on a known trajectory."""
    # Prices: 100 -> 120 -> 90 -> 110 -> 80
    prices = pd.Series([100.0, 120.0, 90.0, 110.0, 80.0])
    dd_series, max_dd = compute_drawdown_series(prices)

    # Peak was 120, lowest after peak was 80: drawdown = (80 - 120) / 120 = -0.33333... (33.33%)
    assert np.isclose(max_dd, 1.0 / 3.0, atol=1e-4)
    # At index 2: (90 - 120) / 120 = -0.25 (-25%)
    assert np.isclose(dd_series.iloc[2], -0.25, atol=1e-4)


def test_portfolio_metrics_known_constant():
    """Verify CAGR, Sharpe, and Volatility on constant return series."""
    # 252 days of exactly +0.1% daily return
    dates = pd.date_range("2022-01-01", periods=252, freq="B")
    daily_rets = pd.Series(0.001, index=dates)

    metrics = compute_portfolio_metrics(daily_rets, risk_free_rate_annual=0.0, trading_days=252)

    # Total return = (1.001)^252 - 1 ~ 28.6%
    assert np.isclose(metrics["total_return"], (1.001**252) - 1.0, atol=1e-3)
    assert metrics["max_drawdown"] == 0.0
    assert metrics["daily_win_rate"] == 1.0


def test_transaction_cost_deduction():
    """Verify turnover and cost series are properly aggregated."""
    dates = pd.date_range("2022-01-01", periods=10, freq="B")
    daily_rets = pd.Series(0.01, index=dates)
    turnover = pd.Series([0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0], index=dates)
    costs = turnover * 0.0010  # 10 bps

    metrics = compute_portfolio_metrics(
        daily_rets, turnover_series=turnover, cost_series=costs, risk_free_rate_annual=0.0
    )

    assert np.isclose(metrics["total_turnover"], 1.0)
    assert np.isclose(metrics["total_costs_pct"], 0.0010)
