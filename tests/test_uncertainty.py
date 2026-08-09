"""
Unit tests for predictive uncertainty estimation and calibration diagnostics.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from src.uncertainty import EnsembleUncertaintyEstimator, evaluate_uncertainty_calibration


def test_ensemble_uncertainty_shapes_and_positivity():
    """Verify uncertainty estimators output correct dimensions and strictly non-negative sigmas."""
    np.random.seed(42)
    n_samples, n_features = 200, 10
    X = np.random.randn(n_samples, n_features)
    y = X[:, 0] * 0.5 + np.random.randn(n_samples) * 0.1

    rf = RandomForestRegressor(n_estimators=15, max_depth=4, random_state=42)
    estimator = EnsembleUncertaintyEstimator(base_estimator=rf, n_bootstrap=10, include_residual_variance=True)
    estimator.fit(X[:150], y[:150])

    X_test = X[150:]
    mu, sigma_ep, sigma_tot = estimator.predict_with_uncertainty(X_test)

    assert len(mu) == len(X_test)
    assert len(sigma_ep) == len(X_test)
    assert len(sigma_tot) == len(X_test)
    assert (sigma_ep >= 0.0).all()
    assert (sigma_tot >= sigma_ep).all()  # Total uncertainty >= epistemic uncertainty


def test_uncertainty_calibration_evaluation():
    """Verify calibration quantile aggregation logic."""
    np.random.seed(42)
    n = 100
    y_true = np.random.randn(n)
    y_pred = y_true + np.random.randn(n) * 0.2
    uncertainty = np.abs(y_true - y_pred) + np.random.uniform(0.01, 0.05, size=n)

    summary, corr, pval = evaluate_uncertainty_calibration(y_true, y_pred, uncertainty, n_quantiles=5)

    assert len(summary) == 5
    assert "mae" in summary.columns
    assert "rmse" in summary.columns
    assert "mean_uncertainty" in summary.columns
    assert -1.0 <= corr <= 1.0
