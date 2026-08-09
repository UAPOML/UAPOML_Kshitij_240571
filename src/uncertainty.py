"""
Predictive Uncertainty Quantification and Calibration Validation module for UAPOML.
Implements ensemble variance, epistemic vs aleatoric decomposition, and error-quantile calibration diagnostics.
"""

from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone

logger = logging.getLogger(__name__)


class EnsembleUncertaintyEstimator:
    """
    Quantifies predictive uncertainty via ensemble disagreement (epistemic uncertainty)
    and residual root mean square error (aleatoric uncertainty).
    """

    def __init__(
        self,
        base_estimator: Any,
        n_bootstrap: int = 20,
        include_residual_variance: bool = True,
        residual_window: int = 60,
        random_state: int = 42,
    ) -> None:
        self.base_estimator = base_estimator
        self.n_bootstrap = n_bootstrap
        self.include_residual_variance = include_residual_variance
        self.residual_window = residual_window
        self.random_state = random_state
        self.models: List[Any] = []
        self.residual_std: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> EnsembleUncertaintyEstimator:
        """
        Train ensemble members. If base_estimator is RandomForest, uses the trained trees directly.
        Otherwise, fits n_bootstrap cloned estimators on bootstrap resamples.
        """
        rng = np.random.RandomState(self.random_state)
        n_samples = len(X)
        self.models = []

        # If base estimator has estimators_ attribute after fit (like RandomForest)
        if hasattr(self.base_estimator, "estimators_") or type(self.base_estimator).__name__ == "RandomForestRegressor":
            model = clone(self.base_estimator)
            model.fit(X, y)
            self.models = [model]
            # Residual std on recent training window (last 250 days)
            recent_idx = slice(-min(250, len(X)), None)
            train_preds = model.predict(X[recent_idx])
            residuals = y[recent_idx] - train_preds
            self.residual_std = float(np.std(residuals))
        else:
            # Bootstrap resample ensemble
            for b in range(self.n_bootstrap):
                indices = rng.choice(n_samples, size=n_samples, replace=True)
                m = clone(self.base_estimator)
                m.fit(X[indices], y[indices])
                self.models.append(m)

            # Residual std across ensemble mean
            recent_idx = slice(-min(250, len(X)), None)
            preds_all = np.array([m.predict(X[recent_idx]) for m in self.models])
            mean_train_pred = np.mean(preds_all, axis=0)
            residuals = y[recent_idx] - mean_train_pred
            self.residual_std = float(np.std(residuals))

        return self

    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates:
        1. mu_hat: Expected predicted return (ensemble mean)
        2. sigma_epistemic: Epistemic uncertainty (model disagreement across ensemble members)
        3. sigma_total: Total predictive uncertainty (including residual variance)
        """
        if len(self.models) == 1 and hasattr(self.models[0], "estimators_"):
            # Extract tree predictions from RandomForest
            rf = self.models[0]
            tree_preds = np.array([tree.predict(X) for tree in rf.estimators_])  # (n_trees, n_samples)
            mu_hat = np.mean(tree_preds, axis=0)
            sigma_epistemic = np.std(tree_preds, axis=0, ddof=1)
        else:
            # Evaluate across bootstrap ensemble members
            preds_all = np.array([m.predict(X) for m in self.models])  # (n_models, n_samples)
            mu_hat = np.mean(preds_all, axis=0)
            sigma_epistemic = np.std(preds_all, axis=0, ddof=1)

        if self.include_residual_variance:
            sigma_total = np.sqrt(sigma_epistemic**2 + self.residual_std**2)
        else:
            sigma_total = sigma_epistemic

        return mu_hat, sigma_epistemic, sigma_total


def evaluate_uncertainty_calibration(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    uncertainty: np.ndarray,
    n_quantiles: int = 5,
) -> Tuple[pd.DataFrame, float, float]:
    """
    Validates whether predictive uncertainty is informative:
    1. Partitions predictions into uncertainty quantiles.
    2. Measures Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) per quantile.
    3. Calculates Spearman correlation between predicted uncertainty and realized absolute error.
    """
    mask = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isnan(uncertainty))
    yt = y_true[mask]
    yp = y_pred[mask]
    unc = uncertainty[mask]

    abs_error = np.abs(yp - yt)
    sq_error = (yp - yt) ** 2

    # Spearman rank correlation between uncertainty and absolute error
    if len(abs_error) > 2 and np.std(unc) > 1e-8:
        corr, pval = spearmanr(unc, abs_error)
        corr, pval = float(corr), float(pval)
    else:
        corr, pval = 0.0, 1.0

    # Quantile binning
    try:
        quantiles = pd.qcut(unc, q=n_quantiles, labels=[f"Q{i+1}" for i in range(n_quantiles)])
    except Exception:
        # Fallback if ties
        quantiles = pd.cut(unc, bins=n_quantiles, labels=[f"Q{i+1}" for i in range(n_quantiles)])

    df_eval = pd.DataFrame({
        "y_true": yt,
        "y_pred": yp,
        "uncertainty": unc,
        "abs_error": abs_error,
        "sq_error": sq_error,
        "quantile": quantiles,
    })

    quantile_summary = df_eval.groupby("quantile", observed=False).agg(
        mean_uncertainty=("uncertainty", "mean"),
        mae=("abs_error", "mean"),
        rmse=("sq_error", lambda x: np.sqrt(np.mean(x))),
        count=("abs_error", "count"),
    ).reset_index()

    return quantile_summary, corr, pval
