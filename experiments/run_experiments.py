"""
Experiment Orchestrator for UAPOML: runs main baseline vs uncertainty comparison,
uncertainty calibration validation, ablation study, and generates all report figures.
"""

from __future__ import annotations
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

# Add root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import yaml

from src.backtest import BacktestResult, WalkForwardBacktestEngine
from src.data import DataLoader, load_universe_data
from src.features import FeatureEngineer, build_and_save_features
from src.metrics import format_metrics_table
from src.uncertainty import evaluate_uncertainty_calibration
from src.visualization import (
    plot_cumulative_equity_curves,
    plot_drawdown_curves,
    plot_portfolio_weights_dynamics,
    plot_prediction_scatter,
    plot_risk_return_comparison,
    plot_rolling_metrics,
    plot_uncertainty_calibration,
)

logger = logging.getLogger("UAPOML_Experiments")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def run_main_experiments(config_path: str = "config.yaml") -> Tuple[Dict[str, BacktestResult], pd.DataFrame, pd.DataFrame]:
    """
    Executes the full core experimental suite:
    1. Data Loading & Feature Generation
    2. Walk-Forward Simulation (Benchmark, EW, Historical MVO, ML-MVO, UAPOML)
    3. Uncertainty Calibration & Validation Diagnostics
    4. Ablation Study across Gamma penalties
    5. Figure and Table Generation
    """
    logger.info("=" * 60)
    logger.info("STARTING UAPOML EXPERIMENTAL SUITE")
    logger.info("=" * 60)

    # 1. Load Data
    loader = DataLoader.from_config(config_path)
    raw_dict = loader.load_all_raw()
    price_df, returns_df = loader.build_aligned_price_matrix(raw_dict)
    bench_returns = loader.get_benchmark_returns(raw_dict)

    # 2. Extract Features
    fe = FeatureEngineer.from_config(config_path)
    panel_df, feature_cols = fe.build_panel_dataset(raw_dict, benchmark_ticker=loader.benchmark)

    logger.info(f"Engineered {len(feature_cols)} features across {len(panel_df)} asset-date rows.")

    # 3. Initialize Walk-Forward Backtest Engine
    engine = WalkForwardBacktestEngine.from_config(
        panel_df=panel_df,
        returns_df=returns_df,
        feature_cols=feature_cols,
        benchmark_returns=bench_returns,
        config_path=config_path,
    )

    # 4. Run Main Simulation with Primary Random Forest Model
    results = engine.run_walk_forward_simulation(model_type="random_forest")

    # 5. Format & Save Main Metrics Table
    metrics_summary = {k: v.metrics for k, v in results.items()}
    metrics_df = format_metrics_table(metrics_summary)
    
    tables_dir = Path("reports/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = Path("reports/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    metrics_df.to_csv(tables_dir / "01_main_strategy_comparison.csv", index=False)
    metrics_df.to_markdown(tables_dir / "01_main_strategy_comparison.md", index=False)
    
    logger.info("\n--- MAIN STRATEGY PERFORMANCE SUMMARY ---")
    logger.info(f"\n{metrics_df.to_string(index=False)}")

    # 6. Uncertainty Calibration Validation
    preds_df = engine.prediction_records_df
    quantile_summary, corr, pval = evaluate_uncertainty_calibration(
        y_true=preds_df["realized_target"].values,
        y_pred=preds_df["mu_hat"].values,
        uncertainty=preds_df["sigma_total"].values,
        n_quantiles=5,
    )
    quantile_summary.to_csv(tables_dir / "02_uncertainty_calibration.csv", index=False)
    quantile_summary.to_markdown(tables_dir / "02_uncertainty_calibration.md", index=False)

    logger.info(f"\n--- UNCERTAINTY CALIBRATION (Rank Corr: {corr:.4f}, p-val: {pval:.2e}) ---")
    logger.info(f"\n{quantile_summary.to_string(index=False)}")

    # 7. Ablation Study: Gamma Penalty Sweeps (gamma in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0])
    logger.info("\n--- RUNNING ABLATION STUDY: UNCERTAINTY PENALTY GAMMA ---")
    gammas = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
    ablation_rows = []

    for g in gammas:
        g_results = engine.run_walk_forward_simulation(model_type="random_forest", uncertainty_gamma=g)
        u_res = g_results["UAPOML_Uncertainty_Aware"]
        m = u_res.metrics
        ablation_rows.append({
            "Gamma (Uncertainty Penalty)": f"{g:.1f}" if g > 0 else "0.0 (No Uncertainty)",
            "Total Return": f"{m.get('total_return', 0)*100:.2f}%",
            "CAGR": f"{m.get('cagr', 0)*100:.2f}%",
            "Annual Volatility": f"{m.get('annual_volatility', 0)*100:.2f}%",
            "Sharpe Ratio": f"{m.get('sharpe_ratio', 0):.2f}",
            "Sortino Ratio": f"{m.get('sortino_ratio', 0):.2f}",
            "Max Drawdown": f"{m.get('max_drawdown', 0)*100:.2f}%",
            "Calmar Ratio": f"{m.get('calmar_ratio', 0):.2f}",
            "Turnover": f"{m.get('total_turnover', 0):.2f}",
        })

    ablation_df = pd.DataFrame(ablation_rows)
    ablation_df.to_csv(tables_dir / "03_ablation_gamma_sensitivity.csv", index=False)
    ablation_df.to_markdown(tables_dir / "03_ablation_gamma_sensitivity.md", index=False)
    logger.info(f"\n{ablation_df.to_string(index=False)}")

    # 8. Generate All Visualizations
    logger.info("\n--- GENERATING PUBLICATION FIGURES ---")
    plot_cumulative_equity_curves(results, str(figures_dir / "01_cumulative_equity_curves.png"))
    plot_drawdown_curves(results, str(figures_dir / "02_drawdown_curves.png"))
    plot_rolling_metrics(results, window=63, save_path=str(figures_dir / "03_rolling_volatility_sharpe.png"))
    if "UAPOML_Uncertainty_Aware" in results:
        plot_portfolio_weights_dynamics(
            results["UAPOML_Uncertainty_Aware"].weights_df,
            "UAPOML Uncertainty-Aware Strategy",
            str(figures_dir / "04_portfolio_weights_dynamics.png"),
        )
    plot_uncertainty_calibration(quantile_summary, corr, str(figures_dir / "05_uncertainty_calibration.png"))
    plot_prediction_scatter(preds_df, str(figures_dir / "06_prediction_vs_realized.png"))
    plot_risk_return_comparison(results, str(figures_dir / "07_risk_return_scatter.png"))

    logger.info("=" * 60)
    logger.info("MAIN EXPERIMENTS COMPLETED SUCCESSFULLY!")
    logger.info("=" * 60)

    return results, metrics_df, ablation_df


if __name__ == "__main__":
    run_main_experiments()
