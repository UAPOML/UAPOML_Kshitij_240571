"""
Interactive Streamlit Dashboard for UAPOML: Uncertainty-Aware Portfolio Optimization.
Provides live interactive exploration of market data, ML predictions, uncertainty calibration,
portfolio weights, and backtest results.
"""

import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="UAPOML: Uncertainty-Aware Portfolio Optimization",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📈 UAPOML: Uncertainty-Aware Portfolio Optimization")
st.markdown(
    """
    **Research Framework:** Evaluating whether incorporating predictive uncertainty from ensemble machine learning 
    models into convex portfolio optimization produces superior risk-adjusted returns over conventional Mean-Variance Optimization.
    """
)

# Sidebar controls
st.sidebar.header("⚙️ Experiment Controls")
tables_dir = Path("reports/tables")
figures_dir = Path("reports/figures")

# Check if precomputed results exist
has_results = (tables_dir / "01_main_strategy_comparison.csv").exists()

if not has_results:
    st.warning("⚠️ No precomputed experiment results found. Please run `python run_pipeline.py` in your terminal first.")
else:
    # Load Main Strategy Comparison
    main_df = pd.read_csv(tables_dir / "01_main_strategy_comparison.csv")
    
    # -------------------------------------------------------------
    # Top KPI Metrics Display
    # -------------------------------------------------------------
    st.subheader("🏆 Strategy Performance Highlights")
    kpi_cols = st.columns(4)

    uapoml_row = main_df[main_df["Strategy"].str.contains("UAPOML", case=False, na=False)]
    spy_row = main_df[main_df["Strategy"].str.contains("Benchmark", case=False, na=False)]

    if not uapoml_row.empty:
        u_sharpe = uapoml_row["Sharpe"].values[0]
        u_cagr = uapoml_row["CAGR"].values[0]
        u_mdd = uapoml_row["Max DD"].values[0]
        u_calmar = uapoml_row["Calmar"].values[0]

        kpi_cols[0].metric("UAPOML Sharpe Ratio", f"{u_sharpe}")
        kpi_cols[1].metric("UAPOML CAGR", f"{u_cagr}")
        kpi_cols[2].metric("UAPOML Max Drawdown", f"{u_mdd}")
        kpi_cols[3].metric("UAPOML Calmar Ratio", f"{u_calmar}")

    # Tabs for detailed navigation
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Strategy Comparison",
        "🎯 Uncertainty Calibration",
        "⚖️ Portfolio Allocation",
        "📉 Risk & Drawdown",
        "🔬 Robustness & Regimes",
    ])

    with tab1:
        st.markdown("### Out-of-Sample Performance Summary")
        st.dataframe(main_df, use_container_width=True)

        fig_path = figures_dir / "01_cumulative_equity_curves.png"
        if fig_path.exists():
            st.image(str(fig_path), caption="Cumulative Portfolio Growth ($100,000 Initial Capital)", use_container_width=True)

        st.markdown("### Risk vs Return Profile")
        fig_rr = figures_dir / "07_risk_return_scatter.png"
        if fig_rr.exists():
            st.image(str(fig_rr), caption="Risk (Annualized Volatility) vs Return (CAGR)", use_container_width=True)

    with tab2:
        st.markdown("### Predictive Uncertainty Validation")
        st.markdown(
            """
            To test whether ML uncertainty is economically meaningful, predictions are partitioned into 5 uncertainty quantiles.
            A valid uncertainty estimator exhibits monotonically increasing forecast error across quantiles.
            """
        )
        if (tables_dir / "02_uncertainty_calibration.csv").exists():
            calib_df = pd.read_csv(tables_dir / "02_uncertainty_calibration.csv")
            st.dataframe(calib_df, use_container_width=True)

        fig_calib = figures_dir / "05_uncertainty_calibration.png"
        if fig_calib.exists():
            st.image(str(fig_calib), caption="Forecast Error by Uncertainty Quantile", use_container_width=True)

        fig_scatter = figures_dir / "06_prediction_vs_realized.png"
        if fig_scatter.exists():
            st.image(str(fig_scatter), caption="Prediction vs Realized Return & Uncertainty vs Error", use_container_width=True)

    with tab3:
        st.markdown("### Dynamic Asset Allocation Over Time")
        fig_weights = figures_dir / "04_portfolio_weights_dynamics.png"
        if fig_weights.exists():
            st.image(str(fig_weights), caption="UAPOML Portfolio Weights (Long-Only, Max 15% Cap)", use_container_width=True)

        if (tables_dir / "03_ablation_gamma_sensitivity.csv").exists():
            st.markdown("### Ablation Study: Uncertainty Penalty ($\gamma$) Sensitivity")
            abl_df = pd.read_csv(tables_dir / "03_ablation_gamma_sensitivity.csv")
            st.dataframe(abl_df, use_container_width=True)

    with tab4:
        st.markdown("### Historical Drawdown & Rolling Volatility")
        fig_dd = figures_dir / "02_drawdown_curves.png"
        if fig_dd.exists():
            st.image(str(fig_dd), caption="Historical Drawdown Profiles", use_container_width=True)

        fig_roll = figures_dir / "03_rolling_volatility_sharpe.png"
        if fig_roll.exists():
            st.image(str(fig_roll), caption="63-Day Rolling Volatility & Sharpe Ratio", use_container_width=True)

    with tab5:
        st.markdown("### Market Regime Breakdown")
        if (tables_dir / "06_regime_analysis.csv").exists():
            reg_df = pd.read_csv(tables_dir / "06_regime_analysis.csv")
            st.dataframe(reg_df, use_container_width=True)

        st.markdown("### Transaction Cost Friction Sensitivity (0 to 30 bps)")
        if (tables_dir / "04_robustness_transaction_costs.csv").exists():
            cost_df = pd.read_csv(tables_dir / "04_robustness_transaction_costs.csv")
            st.dataframe(cost_df, use_container_width=True)
