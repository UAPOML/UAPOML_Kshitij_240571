"""
Robustness analysis module for UAPOML: examines transaction cost sensitivity,
risk aversion lambda sweeps, rebalance frequency variations, and historical market regimes.
"""

from __future__ import annotations
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import yaml

from src.backtest import WalkForwardBacktestEngine
from src.data import DataLoader
from src.features import FeatureEngineer
from src.metrics import compute_portfolio_metrics, format_metrics_table

logger = logging.getLogger("UAPOML_Robustness")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def run_robustness_suite(config_path: str = "config.yaml") -> None:
    """
    Executes comprehensive sensitivity tests across transaction costs, lambda, frequencies, and regimes.
    """
    logger.info("=" * 60)
    logger.info("STARTING UAPOML ROBUSTNESS & REGIME SUITE")
    logger.info("=" * 60)

    tables_dir = Path("reports/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data & Features
    loader = DataLoader.from_config(config_path)
    raw_dict = loader.load_all_raw()
    price_df, returns_df = loader.build_aligned_price_matrix(raw_dict)
    bench_returns = loader.get_benchmark_returns(raw_dict)

    fe = FeatureEngineer.from_config(config_path)
    panel_df, feature_cols = fe.build_panel_dataset(raw_dict, benchmark_ticker=loader.benchmark)

    # Initialize single engine instance to reuse cached walk-forward forecasts
    engine = WalkForwardBacktestEngine.from_config(
        panel_df=panel_df,
        returns_df=returns_df,
        feature_cols=feature_cols,
        benchmark_returns=bench_returns,
        config_path=config_path,
    )

    # ----------------------------------------------------
    # 2. Transaction Cost Sensitivity (0 to 30 bps)
    # ----------------------------------------------------
    logger.info("\n>>> 1. Testing Transaction Cost Sensitivity (0 to 30 bps)...")
    costs_bps = [0.0, 5.0, 10.0, 15.0, 20.0, 30.0]
    cost_rows = []

    for c in costs_bps:
        engine.cost_rate = c / 10000.0
        res = engine.run_walk_forward_simulation(model_type="random_forest")

        for strat in ["Equal_Weight", "MVO_Historical", "ML_MVO_No_Uncertainty", "UAPOML_Uncertainty_Aware"]:
            m = res[strat].metrics
            cost_rows.append({
                "Cost (bps)": int(c),
                "Strategy": strat,
                "CAGR (%)": m.get("cagr", 0) * 100.0,
                "Sharpe": m.get("sharpe_ratio", 0),
                "Max DD (%)": m.get("max_drawdown", 0) * 100.0,
                "Turnover": m.get("total_turnover", 0),
                "Total Costs (%)": m.get("total_costs_pct", 0) * 100.0,
            })

    cost_df = pd.DataFrame(cost_rows)
    cost_df.to_csv(tables_dir / "04_robustness_transaction_costs.csv", index=False)
    cost_df.to_markdown(tables_dir / "04_robustness_transaction_costs.md", index=False)

    # ----------------------------------------------------
    # 3. Risk Aversion Lambda Sensitivity (0.5 to 5.0)
    # ----------------------------------------------------
    logger.info("\n>>> 2. Testing Risk Aversion Lambda Sensitivity (0.5 to 5.0)...")
    lambdas = [0.5, 1.0, 2.0, 3.0, 5.0]
    lambda_rows = []

    # Reset default cost
    engine.cost_rate = 10.0 / 10000.0
    for l_val in lambdas:
        engine.risk_aversion = l_val
        engine.optimizer.risk_aversion = l_val
        res = engine.run_walk_forward_simulation(model_type="random_forest")

        for strat in ["MVO_Historical", "ML_MVO_No_Uncertainty", "UAPOML_Uncertainty_Aware"]:
            m = res[strat].metrics
            lambda_rows.append({
                "Lambda": l_val,
                "Strategy": strat,
                "CAGR (%)": m.get("cagr", 0) * 100.0,
                "Annual Vol (%)": m.get("annual_volatility", 0) * 100.0,
                "Sharpe": m.get("sharpe_ratio", 0),
                "Max DD (%)": m.get("max_drawdown", 0) * 100.0,
            })

    lambda_df = pd.DataFrame(lambda_rows)
    lambda_df.to_csv(tables_dir / "05_robustness_lambda_sensitivity.csv", index=False)
    lambda_df.to_markdown(tables_dir / "05_robustness_lambda_sensitivity.md", index=False)

    # ----------------------------------------------------
    # 4. Market Regime Breakdown
    # ----------------------------------------------------
    logger.info("\n>>> 3. Evaluating Performance across Historical Market Regimes...")
    engine.risk_aversion = 2.0
    engine.optimizer.risk_aversion = 2.0
    main_res = engine.run_walk_forward_simulation(model_type="random_forest")

    regimes = [
        {"name": "2018 Trade War", "start": "2018-01-01", "end": "2018-12-31"},
        {"name": "2020 COVID Shock", "start": "2020-01-01", "end": "2020-12-31"},
        {"name": "2022 Fed Rate Hikes", "start": "2022-01-01", "end": "2022-12-31"},
        {"name": "2023-2024 AI Bull Market", "start": "2023-01-01", "end": "2024-12-31"},
    ]

    regime_rows = []
    for reg in regimes:
        r_name = reg["name"]
        start_d = pd.to_datetime(reg["start"])
        end_d = pd.to_datetime(reg["end"])

        for strat_name, r_obj in main_res.items():
            daily_r = r_obj.daily_returns.loc[start_d:end_d]
            if len(daily_r) < 20:
                continue

            bench_r = bench_returns.loc[daily_r.index] if bench_returns is not None else None
            m = compute_portfolio_metrics(daily_r, benchmark_returns=bench_r, risk_free_rate_annual=0.02)

            regime_rows.append({
                "Regime": r_name,
                "Strategy": strat_name,
                "Total Return": f"{m.get('total_return', 0)*100:.2f}%",
                "CAGR": f"{m.get('cagr', 0)*100:.2f}%",
                "Annual Vol": f"{m.get('annual_volatility', 0)*100:.2f}%",
                "Sharpe": f"{m.get('sharpe_ratio', 0):.2f}",
                "Max DD": f"{m.get('max_drawdown', 0)*100:.2f}%",
            })

    regime_df = pd.DataFrame(regime_rows)
    regime_df.to_csv(tables_dir / "06_regime_analysis.csv", index=False)
    regime_df.to_markdown(tables_dir / "06_regime_analysis.md", index=False)

    logger.info("=" * 60)
    logger.info("ROBUSTNESS & REGIME SUITE COMPLETED SUCCESSFULLY!")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_robustness_suite()
