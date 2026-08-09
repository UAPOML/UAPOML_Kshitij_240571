"""
Publication-quality visualization and diagnostic plotting module for UAPOML.
Generates and saves clean, publication-ready financial and machine learning figures.
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Set clean aesthetic styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 150,
})

STRATEGY_COLORS = {
    "Benchmark_SPY": "#7f7f7f",           # Gray
    "Equal_Weight": "#1f77b4",            # Blue
    "MVO_Historical": "#ff7f0e",          # Orange
    "ML_MVO_No_Uncertainty": "#d62728",    # Red
    "UAPOML_Uncertainty_Aware": "#2ca02c",# Green
}


def plot_cumulative_equity_curves(
    results: Dict[str, Any],
    save_path: Optional[str] = "reports/figures/01_cumulative_equity_curves.png",
) -> plt.Figure:
    """Plot cumulative portfolio equity growth over time ($100,000 initial capital)."""
    fig, ax = plt.subplots(figsize=(12, 6))

    for strat_name, res in results.items():
        color = STRATEGY_COLORS.get(strat_name, None)
        label = f"{strat_name.replace('_', ' ')} (Sharpe: {res.metrics.get('sharpe_ratio', 0):.2f})"
        ax.plot(res.equity_curve.index, res.equity_curve.values, label=label, color=color, linewidth=2.0 if "UAPOML" in strat_name else 1.5)

    ax.set_title("Cumulative Portfolio Value Over Time (Out-of-Sample Walk-Forward)", fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value ($ USD)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))
    ax.legend(loc="upper left", frameon=True)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)
        logger.info(f"Saved figure to {save_path}")

    return fig


def plot_drawdown_curves(
    results: Dict[str, Any],
    save_path: Optional[str] = "reports/figures/02_drawdown_curves.png",
) -> plt.Figure:
    """Plot historical underwater drawdown curves for all strategies."""
    fig, ax = plt.subplots(figsize=(12, 5))

    for strat_name, res in results.items():
        running_max = res.equity_curve.cummax()
        drawdown = (res.equity_curve - running_max) / running_max * 100.0
        color = STRATEGY_COLORS.get(strat_name, None)
        max_dd = res.metrics.get("max_drawdown", 0) * 100.0
        label = f"{strat_name.replace('_', ' ')} (Max DD: -{max_dd:.1f}%)"
        ax.plot(drawdown.index, drawdown.values, label=label, color=color, linewidth=1.5)

    ax.set_title("Underwater Drawdown Profiles (Out-of-Sample)", fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (%)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.0f}%"))
    ax.legend(loc="lower left", frameon=True)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)

    return fig


def plot_rolling_metrics(
    results: Dict[str, Any],
    window: int = 63,
    save_path: Optional[str] = "reports/figures/03_rolling_volatility_sharpe.png",
) -> plt.Figure:
    """Plot 63-day rolling volatility and rolling Sharpe ratios."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    for strat_name, res in results.items():
        color = STRATEGY_COLORS.get(strat_name, None)
        rets = res.daily_returns
        rolling_vol = rets.rolling(window).std() * np.sqrt(252) * 100.0
        rolling_mean = rets.rolling(window).mean() * 252.0
        rolling_sharpe = (rolling_mean - 0.02) / (rets.rolling(window).std() * np.sqrt(252) + 1e-6)

        ax1.plot(rolling_vol.index, rolling_vol.values, label=strat_name.replace("_", " "), color=color, linewidth=1.4)
        ax2.plot(rolling_sharpe.index, rolling_sharpe.values, label=strat_name.replace("_", " "), color=color, linewidth=1.4)

    ax1.set_title("63-Day Rolling Annualized Volatility (%)", fontweight="bold")
    ax1.set_ylabel("Annualized Volatility (%)")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.0f}%"))
    ax1.legend(loc="upper right", frameon=True)
    ax1.grid(True, alpha=0.3)

    ax2.set_title("63-Day Rolling Annualized Sharpe Ratio", fontweight="bold")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Sharpe Ratio")
    ax2.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
    ax2.legend(loc="upper right", frameon=True)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)

    return fig


def plot_portfolio_weights_dynamics(
    weights_df: pd.DataFrame,
    strategy_title: str = "UAPOML Uncertainty-Aware Portfolio",
    save_path: Optional[str] = "reports/figures/04_portfolio_weights_dynamics.png",
) -> plt.Figure:
    """Plot stacked area chart of asset allocation weights over time."""
    fig, ax = plt.subplots(figsize=(12, 6))

    if weights_df.empty:
        return fig

    # Plot stacked area
    cmap = plt.get_cmap("tab20")
    colors = [cmap(i / max(len(weights_df.columns), 1)) for i in range(len(weights_df.columns))]
    ax.stackplot(
        weights_df.index,
        weights_df.T.values,
        labels=weights_df.columns,
        colors=colors,
        alpha=0.85,
    )

    ax.set_title(f"{strategy_title}: Asset Allocation Dynamics", fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Weight")
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x*100:.0f}%"))
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), ncol=1, frameon=True, fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)

    return fig


def plot_uncertainty_calibration(
    quantile_summary: pd.DataFrame,
    corr: float,
    save_path: Optional[str] = "reports/figures/05_uncertainty_calibration.png",
) -> plt.Figure:
    """Plot prediction error vs uncertainty quantiles demonstrating calibration quality."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Bar chart of MAE and RMSE by uncertainty quantile
    x = np.arange(len(quantile_summary))
    width = 0.35
    ax1.bar(x - width/2, quantile_summary["mae"] * 100.0, width, label="MAE (%)", color="#3498db")
    ax1.bar(x + width/2, quantile_summary["rmse"] * 100.0, width, label="RMSE (%)", color="#e74c3c")
    ax1.set_xticks(x)
    ax1.set_xticklabels(quantile_summary["quantile"])
    ax1.set_title(f"Forecast Error by Uncertainty Quantile (Rank Corr: {corr:.3f})", fontweight="bold")
    ax1.set_xlabel("Uncertainty Quantile (Q1 = Lowest, Q5 = Highest)")
    ax1.set_ylabel("Error (%)")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, p: f"{v:.2f}%"))
    ax1.legend(frameon=True)
    ax1.grid(True, alpha=0.3)

    # Line chart of mean uncertainty
    ax2.plot(
        quantile_summary["quantile"],
        quantile_summary["mean_uncertainty"] * 100.0,
        marker="o",
        color="#2ecc71",
        linewidth=2.5,
        markersize=8,
    )
    ax2.set_title("Mean Estimated Uncertainty per Quantile", fontweight="bold")
    ax2.set_xlabel("Uncertainty Quantile")
    ax2.set_ylabel("Mean Predictive Sigma (%)")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, p: f"{v:.2f}%"))
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)

    return fig


def plot_prediction_scatter(
    preds_df: pd.DataFrame,
    save_path: Optional[str] = "reports/figures/06_prediction_vs_realized.png",
) -> plt.Figure:
    """Plot predicted vs realized returns scatter and uncertainty vs absolute error."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    clean = preds_df.dropna(subset=["mu_hat", "realized_target", "sigma_total"])
    if clean.empty:
        return fig

    # 1. Predicted vs Realized
    ax1.scatter(clean["mu_hat"] * 100.0, clean["realized_target"] * 100.0, alpha=0.25, color="#2980b9", s=15)
    # Regression fit line
    m, b = np.polyfit(clean["mu_hat"] * 100.0, clean["realized_target"] * 100.0, 1)
    x_range = np.linspace(clean["mu_hat"].min() * 100.0, clean["mu_hat"].max() * 100.0, 100)
    ax1.plot(x_range, m * x_range + b, color="#e74c3c", linewidth=2, label=f"Fit (Slope: {m:.2f})")
    ax1.set_title("Predicted Return vs Realized 5-Day Return", fontweight="bold")
    ax1.set_xlabel("Predicted 5-Day Return (%)")
    ax1.set_ylabel("Realized 5-Day Return (%)")
    ax1.legend(frameon=True)
    ax1.grid(True, alpha=0.3)

    # 2. Uncertainty vs Absolute Error
    abs_err = np.abs(clean["realized_target"] - clean["mu_hat"]) * 100.0
    ax2.scatter(clean["sigma_total"] * 100.0, abs_err, alpha=0.25, color="#8e44ad", s=15)
    m2, b2 = np.polyfit(clean["sigma_total"] * 100.0, abs_err, 1)
    x_range2 = np.linspace(clean["sigma_total"].min() * 100.0, clean["sigma_total"].max() * 100.0, 100)
    ax2.plot(x_range2, m2 * x_range2 + b2, color="#27ae60", linewidth=2, label=f"Trend (Slope: {m2:.2f})")
    ax2.set_title("Predictive Uncertainty vs Absolute Forecast Error", fontweight="bold")
    ax2.set_xlabel("Predictive Uncertainty Sigma (%)")
    ax2.set_ylabel("Realized Absolute Error (%)")
    ax2.legend(frameon=True)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)

    return fig


def plot_risk_return_comparison(
    results: Dict[str, Any],
    save_path: Optional[str] = "reports/figures/07_risk_return_scatter.png",
) -> plt.Figure:
    """Plot risk-return profile (Annualized Volatility vs CAGR) for all strategies."""
    fig, ax = plt.subplots(figsize=(9, 6))

    for strat_name, res in results.items():
        cagr = res.metrics.get("cagr", 0) * 100.0
        vol = res.metrics.get("annual_volatility", 0) * 100.0
        sharpe = res.metrics.get("sharpe_ratio", 0)
        color = STRATEGY_COLORS.get(strat_name, "#333333")

        ax.scatter(vol, cagr, color=color, s=160, zorder=5, label=f"{strat_name.replace('_', ' ')}")
        ax.annotate(
            f"  {strat_name.replace('_', ' ')}\n  (SR: {sharpe:.2f})",
            (vol, cagr),
            fontsize=10,
            fontweight="bold",
            va="center",
        )

    ax.set_title("Risk vs Return Profile (Out-of-Sample)", fontweight="bold")
    ax.set_xlabel("Annualized Volatility (%)")
    ax.set_ylabel("Compound Annual Growth Rate (CAGR %)")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.1f}%"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, p: f"{y:.1f}%"))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)

    return fig
