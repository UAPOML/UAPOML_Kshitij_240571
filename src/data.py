"""
Data acquisition, caching, validation, and preprocessing module for UAPOML.
"""

from __future__ import annotations
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yaml
import yfinance as yf

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class DataLoader:
    """
    Handles robust downloading, caching, and preprocessing of financial market data.
    Ensures zero look-ahead bias and clean reproducible data pipelines.
    """

    def __init__(
        self,
        universe: Optional[List[str]] = None,
        benchmark: str = "SPY",
        start_date: str = "2015-01-01",
        end_date: str = "2024-12-31",
        raw_data_dir: Union[str, Path] = "data/raw",
        processed_data_dir: Union[str, Path] = "data/processed",
        use_cached: bool = True,
    ) -> None:
        self.universe = universe or [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
            "JPM", "BAC", "UNH", "JNJ", "PFE",
            "XOM", "CVX", "PG", "KO", "HD",
            "COST", "CAT", "HON", "DIS", "BRK-B"
        ]
        self.benchmark = benchmark
        self.start_date = start_date
        self.end_date = end_date
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        self.use_cached = use_cached

        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_config(cls, config_path: str = "config.yaml") -> DataLoader:
        """Instantiate DataLoader directly from a YAML configuration file."""
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        d_cfg = cfg.get("data", {})
        return cls(
            universe=d_cfg.get("universe"),
            benchmark=d_cfg.get("benchmark", "SPY"),
            start_date=d_cfg.get("start_date", "2015-01-01"),
            end_date=d_cfg.get("end_date", "2024-12-31"),
            raw_data_dir=d_cfg.get("raw_data_dir", "data/raw"),
            processed_data_dir=d_cfg.get("processed_data_dir", "data/processed"),
            use_cached=d_cfg.get("use_cached", True),
        )

    def _get_raw_cache_path(self, ticker: str) -> Path:
        clean_ticker = ticker.replace("/", "_").replace("^", "")
        return self.raw_data_dir / f"{clean_ticker}_{self.start_date}_{self.end_date}.parquet"

    def fetch_ticker_data(self, ticker: str, force_download: bool = False) -> pd.DataFrame:
        """
        Fetch OHLCV historical daily data for a single ticker with caching.
        Returns a DataFrame indexed by Date with columns:
        ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Return', 'Log_Return'].
        """
        cache_file = self._get_raw_cache_path(ticker)

        if self.use_cached and not force_download and cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                logger.debug(f"Loaded {ticker} from cache: {cache_file}")
                return df
            except Exception as e:
                logger.warning(f"Failed to read cache for {ticker}, refetching. Error: {e}")

        logger.info(f"Downloading data for {ticker} from Yahoo Finance ({self.start_date} to {self.end_date})...")
        ticker_obj = yf.Ticker(ticker)
        df = ticker_obj.history(start=self.start_date, end=self.end_date, auto_adjust=False)

        if df.empty:
            # Fallback download attempt
            df = yf.download(ticker, start=self.start_date, end=self.end_date, progress=False)

        if df.empty:
            raise ValueError(f"No price data retrieved for ticker: {ticker}")

        # Ensure index is standard DatetimeIndex without timezone
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.index.name = "Date"

        # Standardize column naming
        if "Adj Close" not in df.columns:
            if "Close" in df.columns:
                df["Adj Close"] = df["Close"]
            else:
                raise KeyError(f"Neither 'Adj Close' nor 'Close' found in data for {ticker}")

        # Compute return metrics
        df["Return"] = df["Adj Close"].pct_change()
        df["Log_Return"] = np.log(df["Adj Close"] / df["Adj Close"].shift(1))
        df["Ticker"] = ticker

        # Save to parquet cache
        try:
            df.to_parquet(cache_file)
        except Exception as e:
            logger.warning(f"Could not write cache to {cache_file}: {e}")

        return df

    def load_all_raw(self, force_download: bool = False) -> Dict[str, pd.DataFrame]:
        """Fetch raw daily data for all universe tickers + benchmark."""
        all_tickers = list(self.universe)
        if self.benchmark and self.benchmark not in all_tickers:
            all_tickers.append(self.benchmark)

        data_dict: Dict[str, pd.DataFrame] = {}
        for ticker in all_tickers:
            try:
                df = self.fetch_ticker_data(ticker, force_download=force_download)
                data_dict[ticker] = df
            except Exception as e:
                logger.error(f"Error fetching {ticker}: {e}")
                raise

        return data_dict

    def build_aligned_price_matrix(
        self, data_dict: Optional[Dict[str, pd.DataFrame]] = None, price_col: str = "Adj Close"
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Build aligned (Date x Ticker) matrix for Adjusted Close prices and Daily Returns.
        Ensures strict temporal alignment across all assets.
        """
        if data_dict is None:
            data_dict = self.load_all_raw()

        prices: Dict[str, pd.Series] = {}
        for ticker in self.universe:
            if ticker in data_dict:
                prices[ticker] = data_dict[ticker][price_col]

        price_df = pd.DataFrame(prices).dropna(how="all")
        # Forward fill small missing gaps if any (e.g. trading halt) and drop initial NaNs
        price_df = price_df.ffill().dropna()
        returns_df = price_df.pct_change().dropna()

        return price_df, returns_df

    def get_benchmark_returns(self, data_dict: Optional[Dict[str, pd.DataFrame]] = None) -> pd.Series:
        """Extract daily returns of the benchmark (e.g. SPY)."""
        if data_dict is None:
            data_dict = self.load_all_raw()
        if self.benchmark not in data_dict:
            raise KeyError(f"Benchmark {self.benchmark} not found in fetched data")
        bench_prices = data_dict[self.benchmark]["Adj Close"].dropna()
        return bench_prices.pct_change().dropna().rename(self.benchmark)


def load_universe_data(config_path: str = "config.yaml") -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, Dict[str, pd.DataFrame]]:
    """
    Convenience function to load prices, returns, benchmark returns, and individual raw DataFrames.
    """
    loader = DataLoader.from_config(config_path)
    raw_dict = loader.load_all_raw()
    price_df, returns_df = loader.build_aligned_price_matrix(raw_dict)
    bench_returns = loader.get_benchmark_returns(raw_dict)

    # Re-align index across all
    common_idx = price_df.index.intersection(bench_returns.index)
    price_df = price_df.loc[common_idx]
    returns_df = returns_df.loc[returns_df.index.intersection(common_idx)]
    bench_returns = bench_returns.loc[common_idx]

    return price_df, returns_df, bench_returns, raw_dict
