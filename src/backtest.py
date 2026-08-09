"""
Walk-forward backtesting engine with realistic proportional transaction costs and drift tracking for UAPOML.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yaml

from src.metrics import compute_portfolio_metrics
from src.models import WalkForwardSplitter, instantiate_model
from src.portfolio import PortfolioOptimizer, estimate_covariance_matrix
from src.uncertainty import EnsembleUncertaintyEstimator

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Stores complete time-series and performance metrics from a backtest run."""
    strategy_name: str
    equity_curve: pd.Series
    daily_returns: pd.Series
    weights_df: pd.DataFrame
    turnover_series: pd.Series
    cost_series: pd.Series
    metrics: Dict[str, float]


class WalkForwardBacktestEngine:
    """
    Simulates walk-forward portfolio rebalancing across historical market regimes.
    Enforces strict temporal separation, drift accounting, and proportional transaction costs.
    """

    def __init__(
        self,
        panel_df: pd.DataFrame,
        returns_df: pd.DataFrame,
        feature_cols: List[str],
        tickers: List[str],
        benchmark_returns: Optional[pd.Series] = None,
        train_window_days: int = 756,
        val_window_days: int = 126,
        rebalance_freq_days: int = 5,
        target_horizon_days: int = 5,
        transaction_cost_bps: float = 10.0,
        risk_aversion_lambda: float = 2.0,
        uncertainty_gamma: float = 1.5,
        max_asset_weight: float = 0.15,
        covariance_method: str = "ledoit_wolf",
        cov_lookback_days: int = 126,
        initial_capital: float = 100000.0,
        risk_free_rate_annual: float = 0.02,
    ) -> None:
        self.panel_df = panel_df
        self.returns_df = returns_df
        self.feature_cols = feature_cols
        self.tickers = sorted(tickers)
        self.n_assets = len(self.tickers)
        self.benchmark_returns = benchmark_returns

        self.train_window = train_window_days
        self.val_window = val_window_days
        self.rebalance_freq = rebalance_freq_days
        self.horizon = target_horizon_days
        self.cost_rate = transaction_cost_bps / 10000.0  # convert bps to decimal
        self.risk_aversion = risk_aversion_lambda
        self.uncertainty_gamma = uncertainty_gamma
        self.max_weight = max_asset_weight
        self.cov_method = covariance_method
        self.cov_lookback = cov_lookback_days
        self.initial_capital = initial_capital
        self.rf_annual = risk_free_rate_annual

        self.optimizer = PortfolioOptimizer(
            tickers=self.tickers,
            risk_aversion_lambda=self.risk_aversion,
            uncertainty_gamma=self.uncertainty_gamma,
            max_asset_weight=self.max_weight,
            covariance_method=self.cov_method,
            lookback_days=self.cov_lookback,
        )
        self._forecast_cache: Dict[str, List[Dict[str, Any]]] = {}

    @classmethod
    def from_config(
        cls,
        panel_df: pd.DataFrame,
        returns_df: pd.DataFrame,
        feature_cols: List[str],
        benchmark_returns: Optional[pd.Series] = None,
        config_path: str = "config.yaml",
    ) -> WalkForwardBacktestEngine:
        """Instantiate engine directly from configuration file."""
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        d_cfg = cfg.get("data", {})
        f_cfg = cfg.get("features", {})
        v_cfg = cfg.get("validation", {})
        p_cfg = cfg.get("portfolio", {})
        b_cfg = cfg.get("backtest", {})

        return cls(
            panel_df=panel_df,
            returns_df=returns_df,
            feature_cols=feature_cols,
            tickers=d_cfg.get("universe"),
            benchmark_returns=benchmark_returns,
            train_window_days=v_cfg.get("train_window_days", 756),
            val_window_days=v_cfg.get("val_window_days", 126),
            rebalance_freq_days=b_cfg.get("rebalance_frequency_days", 5),
            target_horizon_days=f_cfg.get("target_horizon_days", 5),
            transaction_cost_bps=b_cfg.get("transaction_cost_bps", 10.0),
            risk_aversion_lambda=p_cfg.get("risk_aversion_lambda", 2.0),
            uncertainty_gamma=p_cfg.get("uncertainty_gamma", 1.5),
            max_asset_weight=p_cfg.get("constraints", {}).get("max_asset_weight", 0.15),
            covariance_method=p_cfg.get("covariance", {}).get("method", "ledoit_wolf"),
            cov_lookback_days=p_cfg.get("covariance", {}).get("lookback_days", 126),
            initial_capital=b_cfg.get("initial_capital", 100000.0),
            risk_free_rate_annual=b_cfg.get("risk_free_rate_annual", 0.02),
        )

    def run_walk_forward_simulation(
        self,
        model_type: str = "random_forest",
        model_params: Optional[Dict[str, Any]] = None,
        uncertainty_gamma: Optional[float] = None,
    ) -> Dict[str, BacktestResult]:
        """
        Runs comprehensive walk-forward backtest across all core strategies simultaneously
        on the exact same out-of-sample periods and market conditions.
        """
        gamma = self.uncertainty_gamma if uncertainty_gamma is None else uncertainty_gamma

        strategies = [
            "Equal_Weight",
            "MVO_Historical",
            "ML_MVO_No_Uncertainty",
            "UAPOML_Uncertainty_Aware",
        ]

        # Tracking state
        history: Dict[str, Dict[str, Any]] = {
            strat: {
                "daily_returns": {},
                "turnovers": {},
                "costs": {},
                "weights": {},
                "current_weights": np.ones(self.n_assets) / self.n_assets,
            }
            for strat in strategies
        }

        # Check if forecasts already cached for this model_type
        if model_type not in self._forecast_cache:
            unique_dates = sorted(self.panel_df["date"].unique())
            splitter = WalkForwardSplitter(
                unique_dates=unique_dates,
                train_window_days=self.train_window,
                test_step_days=self.rebalance_freq,
                target_horizon_days=self.horizon,
                scheme="rolling",
            )
            step_forecasts = []
            all_pred_records = []
            logger.info(f"Computing walk-forward ML predictions & uncertainty with {model_type}...")

            step_count = 0
            for train_dates, test_dates in splitter.split():
                step_count += 1
                rebalance_date = test_dates[0]

                # 1. Slice training panel strictly before rebalance_date - horizon
                train_start_d, train_end_d = train_dates[0], train_dates[-1]
                train_panel = self.panel_df[
                    (self.panel_df["date"] >= train_start_d) & (self.panel_df["date"] <= train_end_d)
                ].dropna(subset=["target"] + self.feature_cols)

                X_train = train_panel[self.feature_cols].values
                y_train = train_panel["target"].values

                # 2. Slice prediction features at rebalance_date
                pred_panel = self.panel_df[self.panel_df["date"] == rebalance_date]
                pred_panel = pred_panel.set_index("ticker").reindex(self.tickers)

                if pred_panel[self.feature_cols].isna().any().any():
                    pred_panel[self.feature_cols] = pred_panel[self.feature_cols].ffill().bfill().fillna(0.0)

                X_pred = pred_panel[self.feature_cols].values

                # 3. Fit base model & uncertainty estimator
                base_model = instantiate_model(model_type, model_params)
                unc_estimator = EnsembleUncertaintyEstimator(
                    base_estimator=base_model,
                    n_bootstrap=20,
                    include_residual_variance=True,
                    random_state=42 + step_count,
                )
                unc_estimator.fit(X_train, y_train)

                # 4. Predict expected return mu and uncertainty sigma
                mu_hat, sigma_epistemic, sigma_total = unc_estimator.predict_with_uncertainty(X_pred)

                # Store predictions for diagnostics
                for idx, tick in enumerate(self.tickers):
                    all_pred_records.append({
                        "date": rebalance_date,
                        "ticker": tick,
                        "mu_hat": mu_hat[idx],
                        "sigma_epistemic": sigma_epistemic[idx],
                        "sigma_total": sigma_total[idx],
                        "realized_target": pred_panel["target"].iloc[idx] if "target" in pred_panel else np.nan,
                    })

                # 5. Estimate Covariance Matrix using historical daily returns up to rebalance_date
                hist_returns = self.returns_df.loc[:rebalance_date, self.tickers]
                cov_matrix = estimate_covariance_matrix(
                    hist_returns, method=self.cov_method, lookback_days=self.cov_lookback
                )
                hist_mu = hist_returns.iloc[-self.cov_lookback:].mean().values * 252.0

                step_forecasts.append({
                    "rebalance_date": rebalance_date,
                    "test_dates": test_dates,
                    "mu_hat": mu_hat,
                    "sigma_total": sigma_total,
                    "cov_matrix": cov_matrix,
                    "hist_mu": hist_mu,
                })

            self._forecast_cache[model_type] = step_forecasts
            self.prediction_records_df = pd.DataFrame(all_pred_records)

        # Replay simulation with target parameters using cached forecasts
        step_forecasts = self._forecast_cache[model_type]
        for item in step_forecasts:
            rebalance_date = item["rebalance_date"]
            test_dates = item["test_dates"]
            mu_hat = item["mu_hat"]
            sigma_total = item["sigma_total"]
            cov_matrix = item["cov_matrix"]
            hist_mu = item["hist_mu"]

            # Scale ML returns from 5-day horizon to annualized scale
            ml_mu_annualized = mu_hat * (252.0 / self.horizon)
            sigma_annualized = sigma_total * np.sqrt(252.0 / self.horizon)

            # Optimize target weights for each strategy
            target_weights: Dict[str, np.ndarray] = {}
            target_weights["Equal_Weight"] = self.optimizer.optimize_equal_weight()
            target_weights["MVO_Historical"] = self.optimizer.optimize_mean_variance(hist_mu, cov_matrix)
            target_weights["ML_MVO_No_Uncertainty"] = self.optimizer.optimize_mean_variance(ml_mu_annualized, cov_matrix)
            target_weights["UAPOML_Uncertainty_Aware"] = self.optimizer.optimize_uncertainty_aware(
                ml_mu_annualized, sigma_annualized, cov_matrix, gamma=gamma
            )

            # 7. Execute Rebalance & simulate daily performance over test_dates
            for strat in strategies:
                w_target = target_weights[strat]
                w_current = history[strat]["current_weights"]

                # Turnover = sum(|w_target - w_current|)
                turnover = float(np.sum(np.abs(w_target - w_current)))
                cost = turnover * self.cost_rate

                history[strat]["turnovers"][rebalance_date] = turnover
                history[strat]["costs"][rebalance_date] = cost
                history[strat]["weights"][rebalance_date] = w_target

                # Simulate daily returns over test_dates holding w_target
                w_active = w_target.copy()
                for day_idx, d in enumerate(test_dates):
                    if d not in self.returns_df.index:
                        continue
                    day_asset_rets = self.returns_df.loc[d, self.tickers].values
                    gross_ret = float(np.dot(w_active, day_asset_rets))

                    # Deduct transaction cost on rebalance day
                    net_ret = gross_ret - (cost if day_idx == 0 else 0.0)
                    history[strat]["daily_returns"][d] = net_ret

                    # Asset drift: w_{i, t+1} = w_{i, t} * (1 + r_{i, t}) / (1 + r_{port, t})
                    w_active = w_active * (1.0 + day_asset_rets)
                    port_sum = np.sum(w_active)
                    if port_sum > 1e-6:
                        w_active = w_active / port_sum

                history[strat]["current_weights"] = w_active

        # 8. Build BacktestResult objects for each strategy
        results: Dict[str, BacktestResult] = {}
        for strat in strategies:
            ret_series = pd.Series(history[strat]["daily_returns"]).sort_index()
            to_series = pd.Series(history[strat]["turnovers"]).sort_index()
            cost_series = pd.Series(history[strat]["costs"]).sort_index()
            weights_df = pd.DataFrame(history[strat]["weights"], index=self.tickers).T.sort_index()

            equity_curve = self.initial_capital * (1.0 + ret_series).cumprod()

            bench_ret = None
            if self.benchmark_returns is not None:
                bench_ret = self.benchmark_returns.loc[self.benchmark_returns.index.intersection(ret_series.index)]

            metrics = compute_portfolio_metrics(
                daily_returns=ret_series,
                benchmark_returns=bench_ret,
                risk_free_rate_annual=self.rf_annual,
                turnover_series=to_series,
                cost_series=cost_series,
            )

            results[strat] = BacktestResult(
                strategy_name=strat,
                equity_curve=equity_curve,
                daily_returns=ret_series,
                weights_df=weights_df,
                turnover_series=to_series,
                cost_series=cost_series,
                metrics=metrics,
            )

        # Benchmark baseline result
        if self.benchmark_returns is not None:
            common_idx = results["Equal_Weight"].daily_returns.index.intersection(self.benchmark_returns.index)
            bench_rets = self.benchmark_returns.loc[common_idx]
            bench_equity = self.initial_capital * (1.0 + bench_rets).cumprod()
            bench_metrics = compute_portfolio_metrics(
                daily_returns=bench_rets,
                benchmark_returns=bench_rets,
                risk_free_rate_annual=self.rf_annual,
            )
            results["Benchmark_SPY"] = BacktestResult(
                strategy_name="Benchmark_SPY",
                equity_curve=bench_equity,
                daily_returns=bench_rets,
                weights_df=pd.DataFrame(),
                turnover_series=pd.Series(0.0, index=bench_rets.index),
                cost_series=pd.Series(0.0, index=bench_rets.index),
                metrics=bench_metrics,
            )

        return results
