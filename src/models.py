"""
Machine Learning models and Walk-Forward time-series validation engine for UAPOML.
Guarantees zero data leakage and enforces chronological out-of-sample evaluation.
"""

from __future__ import annotations
import logging
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def compute_model_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute core quantitative ML evaluation metrics:
    MAE, RMSE, R2, Directional Accuracy (Hit Rate), Information Coefficient (IC).
    """
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_t = y_true[mask]
    y_p = y_pred[mask]

    if len(y_t) == 0:
        return {
            "mae": np.nan, "rmse": np.nan, "r2": np.nan,
            "hit_rate": np.nan, "ic": np.nan, "ic_pvalue": np.nan,
        }

    mae = float(mean_absolute_error(y_t, y_p))
    rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))
    r2 = float(r2_score(y_t, y_p)) if len(y_t) > 1 else 0.0

    # Directional Accuracy: percentage of instances where sign(y_pred) == sign(y_true)
    same_sign = (np.sign(y_t) == np.sign(y_p))
    hit_rate = float(np.mean(same_sign))

    # Information Coefficient (Spearman rank correlation)
    if len(y_t) > 2 and np.std(y_t) > 1e-8 and np.std(y_p) > 1e-8:
        ic, pval = spearmanr(y_p, y_t)
        ic = float(ic) if not np.isnan(ic) else 0.0
        pval = float(pval) if not np.isnan(pval) else 1.0
    else:
        ic, pval = 0.0, 1.0

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "hit_rate": hit_rate,
        "ic": ic,
        "ic_pvalue": pval,
        "n_samples": len(y_t),
    }


class StandardizedRidge(BaseEstimator, RegressorMixin):
    """L2 Ridge Regressor with built-in StandardScaler."""

    def __init__(self, alpha: float = 10.0) -> None:
        self.alpha = alpha
        self.scaler = StandardScaler()
        self.model = Ridge(alpha=self.alpha)

    def fit(self, X: np.ndarray, y: np.ndarray) -> StandardizedRidge:
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)


class WalkForwardSplitter:
    """
    Chronological Walk-Forward Time-Series Splitter.
    Generates (train_indices, test_indices) tuples across time.
    Enforces a gap equal to the target horizon (k days) between train and test
    to prevent target overlap leakage.
    """

    def __init__(
        self,
        unique_dates: List[pd.Timestamp],
        train_window_days: int = 756,  # ~3 years of trading days
        test_step_days: int = 5,       # Step size (rebalance frequency)
        target_horizon_days: int = 5,  # Leakage buffer
        scheme: str = "rolling",       # "rolling" or "expanding"
        min_train_days: int = 504,     # ~2 years
    ) -> None:
        self.dates = sorted(unique_dates)
        self.train_window = train_window_days
        self.test_step = test_step_days
        self.horizon = target_horizon_days
        self.scheme = scheme
        self.min_train = min_train_days

    def split(self) -> Generator[Tuple[List[pd.Timestamp], List[pd.Timestamp]], None, None]:
        """
        Yields (train_dates, test_dates) for each step.
        """
        n_dates = len(self.dates)
        start_test_idx = max(self.train_window + self.horizon, self.min_train + self.horizon)

        for cur_idx in range(start_test_idx, n_dates, self.test_step):
            test_end_idx = min(cur_idx + self.test_step, n_dates)
            test_dates = self.dates[cur_idx:test_end_idx]

            # Training set ends strictly at cur_idx - horizon
            train_end_idx = cur_idx - self.horizon
            if self.scheme == "rolling":
                train_start_idx = max(0, train_end_idx - self.train_window)
            else:  # expanding
                train_start_idx = 0

            train_dates = self.dates[train_start_idx:train_end_idx]
            if len(train_dates) < self.min_train:
                continue

            yield train_dates, test_dates


def instantiate_model(model_type: str, params: Optional[Dict[str, Any]] = None) -> Any:
    """Factory function for initializing regression models."""
    params = params or {}
    if model_type == "ridge":
        return StandardizedRidge(alpha=params.get("alpha", 10.0))
    elif model_type == "random_forest":
        return RandomForestRegressor(
            n_estimators=params.get("n_estimators", 30),
            max_depth=params.get("max_depth", 4),
            min_samples_split=params.get("min_samples_split", 10),
            min_samples_leaf=params.get("min_samples_leaf", 5),
            max_features=params.get("max_features", "sqrt"),
            random_state=params.get("random_state", 42),
            n_jobs=params.get("n_jobs", 1),
        )
    elif model_type == "gradient_boosting":
        return HistGradientBoostingRegressor(
            max_iter=params.get("n_estimators", 100),
            learning_rate=params.get("learning_rate", 0.05),
            max_depth=params.get("max_depth", 4),
            min_samples_leaf=params.get("min_samples_leaf", 5),
            random_state=params.get("random_state", 42),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
