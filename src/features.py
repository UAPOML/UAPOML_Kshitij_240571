"""
Time-series feature engineering module with strict data leakage prevention for UAPOML.
All features at time t are computed using strictly information available at or before t.
"""

from __future__ import annotations
import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def compute_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """Compute Relative Strength Index (RSI) using Wilder's smoothing."""
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Use exponential moving average with alpha = 1/window
    avg_gain = gain.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()

    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def compute_macd(
    prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Compute MACD line, signal line, and histogram."""
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist = macd_line - signal_line
    return macd_line, signal_line, macd_hist


def compute_atr(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14
) -> pd.Series:
    """Compute Average True Range normalized by close price."""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=window, min_periods=window).mean()
    normalized_atr = atr / (close + 1e-10)
    return normalized_atr


class FeatureEngineer:
    """
    Extracts time-series features and target variables across a universe of tickers.
    Guarantees strict temporal alignment without look-ahead bias.
    """

    def __init__(
        self,
        target_horizon_days: int = 5,
        use_log_returns: bool = False,
        return_lags: Optional[List[int]] = None,
        ma_windows: Optional[List[int]] = None,
        vol_windows: Optional[List[int]] = None,
        mom_windows: Optional[List[int]] = None,
        volume_windows: Optional[List[int]] = None,
        rsi_window: int = 14,
        macd_params: Optional[Tuple[int, int, int]] = None,
        atr_window: int = 14,
    ) -> None:
        self.target_horizon = target_horizon_days
        self.use_log_returns = use_log_returns
        self.return_lags = return_lags or [1, 5, 10, 20]
        self.ma_windows = ma_windows or [10, 50, 200]
        self.vol_windows = vol_windows or [5, 20]
        self.mom_windows = mom_windows or [5, 10, 20]
        self.volume_windows = volume_windows or [5, 20]
        self.rsi_window = rsi_window
        self.macd_params = macd_params or (12, 26, 9)
        self.atr_window = atr_window

    @classmethod
    def from_config(cls, config_path: str = "config.yaml") -> FeatureEngineer:
        """Instantiate FeatureEngineer from YAML configuration."""
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        f_cfg = cfg.get("features", {})
        return cls(
            target_horizon_days=f_cfg.get("target_horizon_days", 5),
            use_log_returns=f_cfg.get("use_log_returns", False),
            return_lags=f_cfg.get("return_lags", [1, 5, 10, 20]),
            ma_windows=f_cfg.get("ma_windows", [10, 50, 200]),
            vol_windows=f_cfg.get("vol_windows", [5, 20]),
            mom_windows=f_cfg.get("mom_windows", [5, 10, 20]),
            volume_windows=f_cfg.get("volume_windows", [5, 20]),
            rsi_window=f_cfg.get("rsi_window", 14),
            macd_params=(
                f_cfg.get("macd_fast", 12),
                f_cfg.get("macd_slow", 26),
                f_cfg.get("macd_signal", 9),
            ),
            atr_window=f_cfg.get("atr_window", 14),
        )

    def extract_single_ticker_features(
        self,
        raw_df: pd.DataFrame,
        benchmark_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Build feature set for a single asset.
        Every calculation uses only data at index <= t.
        """
        df = raw_df.copy()
        adj_close = df["Adj Close"]
        volume = df["Volume"] if "Volume" in df.columns else pd.Series(1.0, index=df.index)
        high = df["High"] if "High" in df.columns else adj_close
        low = df["Low"] if "Low" in df.columns else adj_close
        daily_ret = adj_close.pct_change()

        feat = pd.DataFrame(index=df.index)

        # 1. Price / Return Lags
        for lag in self.return_lags:
            feat[f"ret_lag_{lag}d"] = adj_close.pct_change(lag)

        # 2. Moving Average Ratios (Trend)
        for w in self.ma_windows:
            sma = adj_close.rolling(window=w, min_periods=w).mean()
            feat[f"price_to_sma_{w}d"] = (adj_close / sma) - 1.0

        if 10 in self.ma_windows and 50 in self.ma_windows:
            sma10 = adj_close.rolling(10, min_periods=10).mean()
            sma50 = adj_close.rolling(50, min_periods=50).mean()
            feat["sma_10_to_50_ratio"] = (sma10 / sma50) - 1.0

        # 3. Volatility Features
        for w in self.vol_windows:
            feat[f"vol_{w}d"] = daily_ret.rolling(window=w, min_periods=w).std()

        if 5 in self.vol_windows and 20 in self.vol_windows:
            feat["vol_ratio_5_20"] = feat["vol_5d"] / (feat["vol_20d"] + 1e-6)

        # Downside semi-variance (volatility of negative return days over 20 days)
        neg_returns = daily_ret.where(daily_ret < 0, 0.0)
        feat["downside_vol_20d"] = neg_returns.rolling(window=20, min_periods=20).std()

        # 4. Momentum & Rate of Change
        for w in self.mom_windows:
            feat[f"momentum_{w}d"] = (adj_close - adj_close.shift(w)) / (adj_close.shift(w) + 1e-10)

        # 5. Technical Indicators
        feat["rsi_14"] = compute_rsi(adj_close, window=self.rsi_window)
        macd_line, macd_sig, macd_hist = compute_macd(
            adj_close,
            fast=self.macd_params[0],
            slow=self.macd_params[1],
            signal=self.macd_params[2],
        )
        # Normalize MACD by price for cross-asset scale consistency
        feat["macd_line_norm"] = macd_line / (adj_close + 1e-10)
        feat["macd_hist_norm"] = macd_hist / (adj_close + 1e-10)

        if "High" in df.columns and "Low" in df.columns:
            feat["atr_14_norm"] = compute_atr(high, low, adj_close, window=self.atr_window)

        # 6. Volume Dynamics
        if "Volume" in df.columns and (volume > 0).any():
            vol_sma20 = volume.rolling(window=20, min_periods=20).mean()
            vol_sma5 = volume.rolling(window=5, min_periods=5).mean()
            feat["rel_volume_20d"] = (volume / (vol_sma20 + 1e-6)) - 1.0
            feat["vol_ratio_5_20d"] = (vol_sma5 / (vol_sma20 + 1e-6)) - 1.0

        # 7. Benchmark / Market Context Features (if benchmark data provided)
        if benchmark_df is not None and "Adj Close" in benchmark_df.columns:
            bench_close = benchmark_df["Adj Close"].reindex(df.index).ffill()
            bench_ret = bench_close.pct_change()
            feat["bench_ret_5d"] = bench_close.pct_change(5)
            feat["bench_vol_20d"] = bench_ret.rolling(20, min_periods=20).std()
            
            # Rolling 60-day market beta
            cov_60 = daily_ret.rolling(60, min_periods=60).cov(bench_ret)
            var_bench_60 = bench_ret.rolling(60, min_periods=60).var()
            feat["market_beta_60d"] = cov_60 / (var_bench_60 + 1e-6)

        # 8. Target Variable: Future k-day return
        # r_{t, k} = (P_{t+k} - P_t) / P_t
        if self.use_log_returns:
            target = np.log(adj_close.shift(-self.target_horizon) / adj_close)
        else:
            target = (adj_close.shift(-self.target_horizon) - adj_close) / adj_close

        feat["target"] = target
        feat["ticker"] = raw_df["Ticker"].iloc[0] if "Ticker" in raw_df.columns else "UNKNOWN"

        return feat

    def build_panel_dataset(
        self,
        raw_dict: Dict[str, pd.DataFrame],
        benchmark_ticker: str = "SPY",
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Build consolidated multi-asset panel dataset with Date and Ticker index.
        Returns the combined DataFrame and list of feature column names.
        """
        bench_df = raw_dict.get(benchmark_ticker)
        all_features: List[pd.DataFrame] = []

        for ticker, raw_df in raw_dict.items():
            if ticker == benchmark_ticker:
                continue
            try:
                feat_df = self.extract_single_ticker_features(raw_df, benchmark_df=bench_df)
                feat_df["ticker"] = ticker
                feat_df["date"] = feat_df.index
                all_features.append(feat_df)
            except Exception as e:
                logger.error(f"Error extracting features for {ticker}: {e}")

        panel_df = pd.concat(all_features, axis=0).reset_index(drop=True)
        panel_df = panel_df.sort_values(by=["date", "ticker"]).reset_index(drop=True)

        # Identify feature columns (exclude target, ticker, date)
        feature_cols = [
            c for c in panel_df.columns if c not in ["target", "ticker", "date"]
        ]

        return panel_df, feature_cols


def build_and_save_features(
    config_path: str = "config.yaml",
) -> Tuple[pd.DataFrame, List[str]]:
    """Build panel dataset and save processed data for model training."""
    from src.data import DataLoader

    loader = DataLoader.from_config(config_path)
    raw_dict = loader.load_all_raw()

    fe = FeatureEngineer.from_config(config_path)
    panel_df, feature_cols = fe.build_panel_dataset(raw_dict, benchmark_ticker=loader.benchmark)

    out_file = loader.processed_data_dir / "panel_features.parquet"
    panel_df.to_parquet(out_file)
    logger.info(f"Saved {len(panel_df)} panel feature rows to {out_file}")

    return panel_df, feature_cols
