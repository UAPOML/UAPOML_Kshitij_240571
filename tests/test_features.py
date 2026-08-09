"""
Unit tests for time-series feature engineering and data leakage prevention.
"""

import numpy as np
import pandas as pd
import pytest

from src.features import FeatureEngineer, compute_atr, compute_macd, compute_rsi


def create_mock_price_series(n_days: int = 100) -> pd.DataFrame:
    """Generates synthetic deterministic price and volume series."""
    dates = pd.date_range(start="2020-01-01", periods=n_days, freq="B")
    np.random.seed(42)
    daily_rets = np.random.normal(loc=0.0005, scale=0.015, size=n_days)
    prices = 100.0 * np.exp(np.cumsum(daily_rets))
    volumes = np.random.randint(100000, 500000, size=n_days).astype(float)
    highs = prices * (1.0 + np.abs(np.random.normal(0, 0.005, size=n_days)))
    lows = prices * (1.0 - np.abs(np.random.normal(0, 0.005, size=n_days)))

    df = pd.DataFrame({
        "Open": prices * 0.999,
        "High": highs,
        "Low": lows,
        "Close": prices,
        "Adj Close": prices,
        "Volume": volumes,
        "Ticker": "TEST_ASSET",
    }, index=dates)
    return df


def test_rsi_bounds():
    """Verify RSI values remain bounded in [0, 100]."""
    df = create_mock_price_series(100)
    rsi = compute_rsi(df["Adj Close"], window=14).dropna()
    assert (rsi >= 0.0).all()
    assert (rsi <= 100.0).all()


def test_macd_computation():
    """Verify MACD lines and histogram consistency."""
    df = create_mock_price_series(100)
    line, sig, hist = compute_macd(df["Adj Close"], fast=12, slow=26, signal=9)
    assert len(line) == len(df)
    # Check hist = line - sig
    diff = (line - sig) - hist
    assert np.allclose(diff.dropna(), 0.0, atol=1e-8)


def test_no_feature_data_leakage():
    """
    CRITICAL TEST: Verify modifying future price data at t+1..t+N
    does NOT alter the engineered feature values at time t.
    """
    df_original = create_mock_price_series(100)
    fe = FeatureEngineer(target_horizon_days=5)

    feat_orig = fe.extract_single_ticker_features(df_original)

    # Modify prices in the future (from day 70 onwards)
    df_modified = df_original.copy()
    df_modified.iloc[70:, df_modified.columns.get_loc("Adj Close")] *= 2.5
    df_modified.iloc[70:, df_modified.columns.get_loc("High")] *= 2.5
    df_modified.iloc[70:, df_modified.columns.get_loc("Low")] *= 2.5

    feat_mod = fe.extract_single_ticker_features(df_modified)

    feature_cols = [c for c in feat_orig.columns if c not in ["target", "ticker"]]

    # Features at day 69 (before the future modification) MUST be EXACTLY identical
    orig_vals_at_69 = feat_orig.iloc[69][feature_cols].values.astype(float)
    mod_vals_at_69 = feat_mod.iloc[69][feature_cols].values.astype(float)

    assert np.allclose(orig_vals_at_69, mod_vals_at_69, equal_nan=True, atol=1e-8), (
        "Look-ahead leakage detected! Features at time t changed when future prices at t+1 were modified."
    )


def test_target_shift_correctness():
    """Verify that target at index t is exactly (P_{t+k} - P_t) / P_t."""
    df = create_mock_price_series(100)
    fe = FeatureEngineer(target_horizon_days=5)
    feat = fe.extract_single_ticker_features(df)

    p = df["Adj Close"].values
    t = 40
    expected_target = (p[t + 5] - p[t]) / p[t]
    actual_target = feat["target"].iloc[t]

    assert np.isclose(expected_target, actual_target, atol=1e-7)
