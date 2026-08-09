"""
Unit tests for portfolio optimizers and weight constraint enforcement.
"""

import numpy as np
import pandas as pd
import pytest

from src.portfolio import PortfolioOptimizer, estimate_covariance_matrix


def test_equal_weight_constraints():
    """Verify equal weight returns uniform weights summing to 1."""
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN"]
    opt = PortfolioOptimizer(tickers=tickers)
    w = opt.optimize_equal_weight()
    assert len(w) == 4
    assert np.allclose(w, 0.25)
    assert np.isclose(np.sum(w), 1.0)


def test_mean_variance_constraints():
    """Verify MVO respects sum=1, non-negativity, and max asset cap."""
    tickers = ["A", "B", "C", "D", "E"]
    n = len(tickers)
    opt = PortfolioOptimizer(tickers=tickers, max_asset_weight=0.30, min_asset_weight=0.0)

    # Asset A has massive expected return, but should still be capped at max_weight (0.30)
    mu = np.array([0.50, 0.05, 0.04, 0.03, 0.02])
    cov = np.eye(n) * 0.04  # 20% annual volatility each

    w = opt.optimize_mean_variance(mu, cov)

    assert np.isclose(np.sum(w), 1.0, atol=1e-5), f"Weights sum to {np.sum(w)}"
    assert (w >= -1e-7).all(), f"Negative weights detected: {w}"
    assert (w <= 0.30 + 1e-5).all(), f"Max weight violated: {w.max()} > 0.30"


def test_uncertainty_aware_penalty():
    """
    Verify that increasing uncertainty penalty gamma reduces allocation to uncertain assets.
    """
    tickers = ["HighReturn_HighUnc", "ModerateReturn_LowUnc", "LowReturn_LowUnc"]
    opt = PortfolioOptimizer(tickers=tickers, max_asset_weight=0.80, min_asset_weight=0.0)

    mu = np.array([0.25, 0.20, 0.05])
    # High return asset has 5x the uncertainty
    sigma = np.array([0.30, 0.02, 0.02])
    cov = np.eye(3) * 0.04

    # Without uncertainty penalty (gamma = 0)
    w_no_unc = opt.optimize_uncertainty_aware(mu, sigma, cov, gamma=0.0)
    # With strong uncertainty penalty (gamma = 3.0)
    w_with_unc = opt.optimize_uncertainty_aware(mu, sigma, cov, gamma=3.0)

    # The weight in the highly uncertain asset (index 0) must decrease
    assert w_with_unc[0] < w_no_unc[0], (
        f"Uncertainty penalty failed: weight_with_unc={w_with_unc[0]} vs weight_no_unc={w_no_unc[0]}"
    )


def test_covariance_positive_definiteness():
    """Verify Ledoit-Wolf covariance matrix is positive definite."""
    np.random.seed(42)
    n_days, n_assets = 200, 5
    rets = pd.DataFrame(np.random.randn(n_days, n_assets) * 0.01)
    cov = estimate_covariance_matrix(rets, method="ledoit_wolf", lookback_days=100)

    # Eigenvalues must all be strictly positive
    eigenvalues = np.linalg.eigvalsh(cov)
    assert (eigenvalues > 0).all(), f"Negative eigenvalue found: {eigenvalues}"
