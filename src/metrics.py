"""
Financial performance and risk metrics calculation module for UAPOML.
Computes CAGR, Volatility, Sharpe, Sortino, Calmar, Maximum Drawdown, Turnover, and Benchmark metrics.
"""

from __future__ import annotations
import logging
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_drawdown_series(equity_curve: pd.Series) -> Tuple[pd.Series, float]:
    """
    Computes daily drawdown series and maximum drawdown (MDD).
    Returns (drawdown_series, max_drawdown).
    """
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / (running_max + 1e-10)
    max_dd = float(np.abs(drawdown.min()))
    return drawdown, max_dd


def compute_portfolio_metrics(
    daily_returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    risk_free_rate_annual: float = 0.02,
    trading_days: int = 252,
    turnover_series: Optional[pd.Series] = None,
    cost_series: Optional[pd.Series] = None,
) -> Dict[str, float]:
    """
    Computes comprehensive financial performance and risk metrics for a backtested portfolio.
    """
    rets = daily_returns.dropna()
    n_days = len(rets)

    if n_days == 0:
        return {}

    # Cumulative Return & CAGR
    equity_curve = (1.0 + rets).cumprod()
    total_return = float(equity_curve.iloc[-1] - 1.0)
    years = n_days / float(trading_days)
    cagr = float((1.0 + total_return) ** (1.0 / max(years, 0.01)) - 1.0) if total_return > -1.0 else -1.0

    # Volatility
    daily_vol = float(rets.std())
    annual_vol = daily_vol * np.sqrt(trading_days)

    # Sharpe Ratio
    rf_daily = risk_free_rate_annual / trading_days
    excess_returns = rets - rf_daily
    excess_mean = float(excess_returns.mean()) * trading_days
    sharpe = float(excess_mean / (annual_vol + 1e-8)) if annual_vol > 1e-8 else 0.0

    # Downside Volatility & Sortino Ratio
    neg_excess = excess_returns.loc[excess_returns < 0]
    downside_vol_daily = float(np.sqrt(np.mean(neg_excess**2))) if len(neg_excess) > 0 else 1e-8
    downside_vol_annual = downside_vol_daily * np.sqrt(trading_days)
    sortino = float(excess_mean / (downside_vol_annual + 1e-8)) if downside_vol_annual > 1e-8 else 0.0

    # Maximum Drawdown & Calmar Ratio
    _, max_dd = compute_drawdown_series(equity_curve)
    calmar = float(cagr / max_dd) if max_dd > 1e-6 else 0.0

    # Turnover & Costs
    total_turnover = float(turnover_series.sum()) if turnover_series is not None else 0.0
    avg_turnover = float(turnover_series.mean()) if turnover_series is not None else 0.0
    total_costs = float(cost_series.sum()) if cost_series is not None else 0.0

    # Information Ratio & Beta (if benchmark available)
    ir = 0.0
    beta = 0.0
    alpha = 0.0
    if benchmark_returns is not None:
        common_idx = rets.index.intersection(benchmark_returns.index)
        if len(common_idx) > 10:
            p_ret = rets.loc[common_idx]
            b_ret = benchmark_returns.loc[common_idx]
            active_ret = p_ret - b_ret
            tracking_error = float(active_ret.std()) * np.sqrt(trading_days)
            ir = float((active_ret.mean() * trading_days) / (tracking_error + 1e-8))

            # Beta and Alpha (CAPM)
            cov = float(np.cov(p_ret, b_ret)[0, 1])
            b_var = float(np.var(b_ret))
            beta = float(cov / (b_var + 1e-10)) if b_var > 1e-10 else 0.0
            alpha = float((p_ret.mean() * trading_days) - (rf_daily * trading_days + beta * (b_ret.mean() * trading_days - rf_daily * trading_days)))

    # Win Rate
    win_rate = float(np.mean(rets > 0))

    return {
        "total_return": total_return,
        "cagr": cagr,
        "annual_volatility": annual_vol,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_dd,
        "calmar_ratio": calmar,
        "total_turnover": total_turnover,
        "avg_turnover_per_rebalance": avg_turnover,
        "total_costs_pct": total_costs,
        "information_ratio": ir,
        "beta": beta,
        "alpha": alpha,
        "daily_win_rate": win_rate,
        "n_trading_days": n_days,
    }


def format_metrics_table(metrics_dict: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Formats multi-strategy metrics into a presentation-ready DataFrame.
    """
    rows = []
    for strat_name, m in metrics_dict.items():
        rows.append({
            "Strategy": strat_name,
            "Total Return": f"{m.get('total_return', 0)*100:.2f}%",
            "CAGR": f"{m.get('cagr', 0)*100:.2f}%",
            "Annual Vol": f"{m.get('annual_volatility', 0)*100:.2f}%",
            "Sharpe": f"{m.get('sharpe_ratio', 0):.2f}",
            "Sortino": f"{m.get('sortino_ratio', 0):.2f}",
            "Max DD": f"{m.get('max_drawdown', 0)*100:.2f}%",
            "Calmar": f"{m.get('calmar_ratio', 0):.2f}",
            "Turnover": f"{m.get('total_turnover', 0):.2f}",
            "IR": f"{m.get('information_ratio', 0):.2f}",
        })
    return pd.DataFrame(rows)
