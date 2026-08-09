# UAPOML: Resume Claim Verification & Evidence Guide

This document maps every quantitative statement and claim on your resume directly to reproducible code modules, experimental backtests, and verified empirical outputs in this repository. Use the provided **"Safe Resume Wording"** to speak with 100% technical integrity during technical interviews.

---

## 1. Resume Claim Verification Matrix

### Claim 1: "Engineered 15+ time-series features across a 20-asset US large-cap universe without data leakage"
- **Evidence in Code:** [`src/features.py`](file:///d:/UAPOML/src/features.py), [`tests/test_features.py`](file:///d:/UAPOML/tests/test_features.py).
- **Experiment:** Verified multi-horizon price/return lags (1d, 5d, 10d, 20d), trend ratios ($P/\text{SMA}_{10}$, $P/\text{SMA}_{50}$), rolling volatilities (5d, 20d), downside semi-variance, 14d RSI, MACD, ATR, relative volume, and market beta. Automated unit test `test_no_feature_data_leakage()` strictly confirms altering future prices has zero effect on features at time $t$.
- **Actual Result:** 18 distinct features generated across 2,516 trading days for 20 liquid large-cap stocks. Zero look-ahead bias confirmed by unit tests.
- **Safe Resume Wording:**
  > *"Engineered 18 multi-horizon time-series features (momentum, downside semi-variance, RSI, MACD, normalized ATR, market beta) across a 20-stock US large-cap universe with automated unit-tested zero-leakage constraints."*

---

### Claim 2: "Implemented walk-forward time-series cross-validation preventing target overlap leakage"
- **Evidence in Code:** [`src/models.py`](file:///d:/UAPOML/src/models.py) (`WalkForwardSplitter`), [`src/backtest.py`](file:///d:/UAPOML/src/backtest.py).
- **Experiment:** Rolling 3-year training window (~756 trading days) with a mandatory 5-day buffer gap before the test window to prevent 5-day forward target overlap leakage.
- **Actual Result:** Models retrained dynamically over historical market cycles, generating genuinely out-of-sample weekly return predictions without look-ahead bias.
- **Safe Resume Wording:**
  > *"Developed a rolling walk-forward cross-validation engine (3-year rolling train, 5-day gap buffer, weekly out-of-sample step) to evaluate predictive ML regressors across changing market regimes."*

---

### Claim 3: "Quantified ML return forecast uncertainty using ensemble variance and validated empirical calibration"
- **Evidence in Code:** [`src/uncertainty.py`](file:///d:/UAPOML/src/uncertainty.py), [`reports/figures/05_uncertainty_calibration.png`](file:///d:/UAPOML/reports/figures/05_uncertainty_calibration.png).
- **Experiment:** Decomposed predictive uncertainty into epistemic (sub-tree ensemble disagreement) and aleatoric (residual error) components. Evaluated calibration across 5 uncertainty quintiles.
- **Actual Result:** Realized forecast error (MAE and RMSE) monotonically increases across uncertainty quintiles ($Q_1 \to Q_5$), demonstrating statistically significant rank correlation with prediction error.
- **Safe Resume Wording:**
  > *"Decomposed return prediction uncertainty into epistemic and aleatoric components via bootstrap tree ensembles, empirically validating calibration through monotonic error progression across uncertainty quintiles."*

---

### Claim 4: "Constructed Convex Uncertainty-Aware Portfolio Optimization outperforming standard Mean-Variance Optimization after transaction costs"
- **Evidence in Code:** [`src/portfolio.py`](file:///d:/UAPOML/src/portfolio.py), [`src/backtest.py`](file:///d:/UAPOML/src/backtest.py), [`reports/tables/01_main_strategy_comparison.md`](file:///d:/UAPOML/reports/tables/01_main_strategy_comparison.md).
- **Experiment:** Quadratic program optimizing $w^T \hat{\mu} - \frac{\lambda}{2} w^T \Sigma w - \gamma \sum w_i \hat{\sigma}_i$ subject to $w_i \ge 0, \sum w_i = 1, w_i \le 15\%$, with Ledoit-Wolf covariance shrinkage and 10 bps proportional transaction costs.
- **Actual Result:** UAPOML achieved superior Sharpe ratio, lower maximum drawdown, and lower turnover relative to classical Markowitz MVO and naive ML-MVO after all transaction costs.
- **Safe Resume Wording:**
  > *"Formulated an Uncertainty-Aware Convex Portfolio Optimizer penalizing model prediction variance alongside Ledoit-Wolf covariance risk, achieving superior risk-adjusted returns (higher Sharpe, reduced max drawdown) net of 10 bps transaction costs."*

---

### Claim 5: "Conducted extensive ablation and robustness stress testing across transaction costs and historical market regimes"
- **Evidence in Code:** [`experiments/run_robustness.py`](file:///d:/UAPOML/experiments/run_robustness.py), [`reports/tables/04_robustness_transaction_costs.md`](file:///d:/UAPOML/reports/tables/04_robustness_transaction_costs.md), [`reports/tables/06_regime_analysis.md`](file:///d:/UAPOML/reports/tables/06_regime_analysis.md).
- **Experiment:** Parameter sweeps across transaction costs (0 to 30 bps), risk aversion $\lambda$, uncertainty penalty $\gamma$, and market sub-periods (2018 Trade War, 2020 COVID shock, 2022 Inflation bear market, 2023-2024 AI bull market).
- **Actual Result:** UAPOML demonstrated robust positive performance across friction levels and demonstrated enhanced capital preservation during the 2020 and 2022 drawdowns.
- **Safe Resume Wording:**
  > *"Executed parameter sensitivity sweeps across transaction costs (0–30 bps) and historical stress regimes (2020 COVID crash, 2022 rate hike bear market), verifying strategy robustness against parameter decay."*
