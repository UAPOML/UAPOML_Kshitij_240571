"""
Portfolio construction and convex optimization module for UAPOML.
Implements Equal Weight, Mean-Variance Optimization, and Uncertainty-Aware Portfolio Optimization.
"""

from __future__ import annotations
import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

logger = logging.getLogger(__name__)


def estimate_covariance_matrix(
    returns_df: pd.DataFrame,
    method: str = "ledoit_wolf",
    lookback_days: int = 126,
) -> np.ndarray:
    """
    Estimates the asset return covariance matrix over a historical lookback window.
    Uses Ledoit-Wolf shrinkage to ensure positive definiteness and reduce estimation error.
    """
    recent_returns = returns_df.iloc[-lookback_days:].values
    if method == "ledoit_wolf":
        try:
            lw = LedoitWolf()
            cov = lw.fit(recent_returns).covariance_
        except Exception:
            cov = np.cov(recent_returns, rowvar=False)
    else:
        cov = np.cov(recent_returns, rowvar=False)

    # Annualize covariance (252 trading days)
    cov_annualized = cov * 252.0
    # Add minimal diagonal ridge to guarantee positive definiteness
    cov_annualized += np.eye(cov_annualized.shape[0]) * 1e-6
    return cov_annualized


class PortfolioOptimizer:
    """
    Convex portfolio optimizer supporting multiple asset allocation paradigms:
    - Equal Weight Baseline
    - Classical Mean-Variance Optimization (Historical & ML expected returns)
    - Uncertainty-Aware Portfolio Optimization (penalizing predictive uncertainty)
    - Risk-Adjusted Score Weighted Allocation
    """

    def __init__(
        self,
        tickers: List[str],
        risk_aversion_lambda: float = 2.0,
        uncertainty_gamma: float = 1.5,
        max_asset_weight: float = 0.15,
        min_asset_weight: float = 0.0,
        covariance_method: str = "ledoit_wolf",
        lookback_days: int = 126,
        score_epsilon: float = 1e-4,
    ) -> None:
        self.tickers = tickers
        self.n_assets = len(tickers)
        self.risk_aversion = risk_aversion_lambda
        self.uncertainty_gamma = uncertainty_gamma
        self.max_weight = max_asset_weight
        self.min_weight = min_asset_weight
        self.cov_method = covariance_method
        self.lookback = lookback_days
        self.score_epsilon = score_epsilon

    def optimize_equal_weight(self) -> np.ndarray:
        """Constructs an Equal Weight portfolio: w_i = 1 / N."""
        return np.ones(self.n_assets) / self.n_assets

    def optimize_mean_variance(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
    ) -> np.ndarray:
        """
        Solves standard Mean-Variance Optimization:
            max w^T mu - 0.5 * lambda * w^T Sigma w
            s.t. sum(w) = 1, min_w <= w_i <= max_w
        """
        n = self.n_assets
        init_w = np.ones(n) / n
        bounds = [(self.min_weight, self.max_weight) for _ in range(n)]
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        def objective(w: np.ndarray) -> float:
            port_ret = np.dot(w, expected_returns)
            port_var = np.dot(w, np.dot(cov_matrix, w))
            # Negative for minimization
            return -(port_ret - 0.5 * self.risk_aversion * port_var)

        def objective_grad(w: np.ndarray) -> np.ndarray:
            return -(expected_returns - self.risk_aversion * np.dot(cov_matrix, w))

        res = minimize(
            objective,
            init_w,
            jac=objective_grad,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 100, "ftol": 1e-6},
        )

        if not res.success:
            logger.warning(f"MVO optimization did not converge: {res.message}. Using equal weights.")
            return self.optimize_equal_weight()

        # Clean small numerical noise
        w_opt = np.maximum(0.0, res.x)
        return w_opt / np.sum(w_opt)

    def optimize_uncertainty_aware(
        self,
        expected_returns: np.ndarray,
        uncertainties: np.ndarray,
        cov_matrix: np.ndarray,
        gamma: Optional[float] = None,
    ) -> np.ndarray:
        """
        Solves Uncertainty-Aware Portfolio Optimization:
            max w^T mu - 0.5 * lambda * w^T Sigma w - gamma * sum(w_i * sigma_i)
            s.t. sum(w) = 1, min_w <= w_i <= max_w
        Equivalently:
            min 0.5 * lambda * w^T Sigma w - w^T (mu - gamma * sigma)
        """
        n = self.n_assets
        g = self.uncertainty_gamma if gamma is None else gamma
        init_w = np.ones(n) / n
        bounds = [(self.min_weight, self.max_weight) for _ in range(n)]
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        # Effective expected return adjusted by uncertainty penalty
        adjusted_mu = expected_returns - g * uncertainties

        def objective(w: np.ndarray) -> float:
            port_ret_adj = np.dot(w, adjusted_mu)
            port_var = np.dot(w, np.dot(cov_matrix, w))
            return -(port_ret_adj - 0.5 * self.risk_aversion * port_var)

        def objective_grad(w: np.ndarray) -> np.ndarray:
            return -(adjusted_mu - self.risk_aversion * np.dot(cov_matrix, w))

        res = minimize(
            objective,
            init_w,
            jac=objective_grad,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 100, "ftol": 1e-6},
        )

        if not res.success:
            logger.warning(f"UAPOML optimization did not converge: {res.message}. Using equal weights.")
            return self.optimize_equal_weight()

        w_opt = np.maximum(0.0, res.x)
        return w_opt / np.sum(w_opt)

    def compute_risk_adjusted_scores(
        self,
        expected_returns: np.ndarray,
        uncertainties: np.ndarray,
        formulation: str = "ratio",
    ) -> np.ndarray:
        """
        Computes risk-adjusted signal scores:
        - Ratio: score_i = mu_i / (sigma_i + epsilon)
        - Penalty: score_i = mu_i - gamma * sigma_i
        """
        if formulation == "ratio":
            return expected_returns / (uncertainties + self.score_epsilon)
        elif formulation == "penalty":
            return expected_returns - self.uncertainty_gamma * uncertainties
        else:
            raise ValueError(f"Unknown score formulation: {formulation}")
