"""
Main CLI Orchestration Entrypoint for UAPOML:
Executes full data pipeline, feature engineering, walk-forward models, uncertainty estimation,
portfolio optimization, backtesting, ablation, and robustness suites.
"""

from __future__ import annotations
import logging
import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import yaml
from experiments.run_experiments import run_main_experiments
from experiments.run_robustness import run_robustness_suite

logger = logging.getLogger("UAPOML_Pipeline")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main() -> None:
    start_time = time.time()
    logger.info("================================================================")
    logger.info("   UAPOML: Uncertainty-Aware Portfolio Optimization with ML     ")
    logger.info("================================================================")

    config_path = "config.yaml"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file {config_path} not found.")

    # 1. Run Main Experiments (Baselines, ML, Uncertainty, Ablations, Figures)
    logger.info("\n>>> STEP 1: Running Main Strategy Experiments & Diagnostics...")
    results, metrics_df, ablation_df = run_main_experiments(config_path)

    # 2. Run Robustness and Regime Analyses
    logger.info("\n>>> STEP 2: Running Robustness & Multi-Regime Analyses...")
    run_robustness_suite(config_path)

    elapsed = time.time() - start_time
    logger.info("================================================================")
    logger.info(f"   PIPELINE COMPLETED SUCCESSFULLY IN {elapsed:.1f} SECONDS    ")
    logger.info("   Outputs saved to:                                           ")
    logger.info("     - Reports: reports/final_report.md                        ")
    logger.info("     - Tables:  reports/tables/                                ")
    logger.info("     - Figures: reports/figures/                               ")
    logger.info("================================================================")


if __name__ == "__main__":
    main()
