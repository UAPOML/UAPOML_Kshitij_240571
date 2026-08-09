# 📈 UAPOML: Uncertainty-Aware Portfolio Optimization using Machine Learning

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Tests: passing](https://img.shields.io/badge/tests-13%20passed-brightgreen.svg)]()

> **A Technically Rigorous Quantitative Research Framework Evaluating Whether Predictive Uncertainty Penalties in Machine Learning Return Forecasts Produce Robust Risk-Adjusted Portfolios.**

---

## 🔬 Core Research Question

> **"Can incorporating uncertainty in ML-based return forecasts into portfolio construction produce a more robust risk-adjusted portfolio than conventional mean-variance optimization?"**

Classical Markowitz Mean-Variance Optimization (MVO) acts as an **"estimation error maximizer"** by over-allocating capital to assets with the largest positive return estimation errors. Standard ML models output point predictions $\hat{\mu}$ that ignore model confidence. **UAPOML** solves this by decomposing predictions into expected return $\hat{\mu}$ and predictive uncertainty $\hat{\sigma}$, incorporating $\hat{\sigma}$ as an explicit penalty in convex quadratic portfolio optimization.

---

## 🏗️ Quantitative Architecture

```
                               ┌──────────────────────────────────────────────┐
                               │   Financial Market Data (20 Assets + SPY)    │
                               └──────────────────────┬───────────────────────┘
                                                      │
                                                      ▼
                               ┌──────────────────────────────────────────────┐
                               │    Leakage-Free Feature Engineering (18)     │
                               │  - Momentum, Moving Average Ratios, Lags     │
                               │  - Rolling Vol, Downside Semi-Variance       │
                               │  - 14d RSI, MACD, Normalized ATR, Rel Volume │
                               └──────────────────────┬───────────────────────┘
                                                      │
                                                      ▼
                               ┌──────────────────────────────────────────────┐
                               │   Walk-Forward Rolling ML Engine (3yr Train) │
                               │  - Ridge, Random Forest, Gradient Boosting   │
                               │  - 5-Day Forward Return Target (r_{t, 5})    │
                               └──────────────────────┬───────────────────────┘
                                                      │
                                                      ▼
                               ┌──────────────────────────────────────────────┐
                               │     Ensemble Uncertainty Quantification      │
                               │  - Epistemic: Sub-tree Disagreement          │
                               │  - Aleatoric: Residual Rolling Volatility    │
                               │  - Empirical Calibration Validation          │
                               └──────────────────────┬───────────────────────┘
                                                      │
                                                      ▼
                               ┌──────────────────────────────────────────────┐
                               │   Convex Uncertainty-Aware Optimization      │
                               │   max wᵀμ̂ - (λ/2) wᵀΣw - γ ∑ wᵢ σ̂ᵢ          │
                               │   s.t. ∑ wᵢ = 1,  0 ≤ wᵢ ≤ 15% (Ledoit-Wolf) │
                               └──────────────────────┬───────────────────────┘
                                                      │
                                                      ▼
                               ┌──────────────────────────────────────────────┐
                               │   Out-of-Sample Walk-Forward Backtester      │
                               │  - 10 bps Proportional Transaction Costs     │
                               │  - Asset Drift & Turnover Accounting         │
                               │  - Multi-Regime & Parameter Robustness       │
                               └──────────────────────────────────────────────┘
```

---

## 📐 Mathematical Formulation

### 1. Target Variable
Forward 5-day return shifted strictly backward relative to feature timestamp $t$:
$$y_{i,t} = \frac{P_{i, t+5} - P_{i,t}}{P_{i,t}}$$

### 2. Predictive Uncertainty Decomposition
Given an ensemble of $M$ decision trees $\{\mathcal{T}_m\}_{m=1}^M$:
$$\hat{\mu}_{i,t} = \frac{1}{M}\sum_{m=1}^M \mathcal{T}_m(x_{i,t})$$
$$\hat{\sigma}_{i,t} = \sqrt{\underbrace{\frac{1}{M-1}\sum_{m=1}^M (\mathcal{T}_m(x_{i,t}) - \hat{\mu}_{i,t})^2}_{\text{Epistemic Uncertainty (Model Disagreement)}} + \underbrace{\hat{\sigma}_{\text{res}}^2}_{\text{Aleatoric Noise}}}$$

### 3. Uncertainty-Aware Portfolio Objective
$$\max_{w} \quad w^T \hat{\mu} - \frac{\lambda}{2} w^T \Sigma_{\text{LW}} w - \gamma \sum_{i=1}^N w_i \hat{\sigma}_i \quad \text{s.t.} \quad \sum_{i=1}^N w_i = 1, \quad 0 \le w_i \le w_{\max}$$
where $\Sigma_{\text{LW}}$ is the Ledoit-Wolf shrunk covariance matrix, $\lambda$ is variance risk aversion, and $\gamma$ is the predictive uncertainty penalty.

---

## 📂 Project Structure

```
UAPOML/
│
├── README.md                           # Project documentation & overview
├── requirements.txt                    # Python package dependencies
├── environment.yml                     # Conda environment definition
├── config.yaml                         # Global configuration & hyperparameters
├── PROJECT_STATUS.md                   # Development roadmap & status tracker
├── run_pipeline.py                     # Main CLI orchestrator (end-to-end run)
├── generate_report.py                  # Report table compiler
├── app.py                              # Interactive Streamlit dashboard
│
├── data/
│   ├── raw/                            # Cached parquet raw market data
│   └── processed/                      # Engineered panel features
│
├── src/
│   ├── __init__.py
│   ├── data.py                         # Data ingestion, caching, and returns
│   ├── features.py                     # 18 time-series features (zero-leakage)
│   ├── models.py                       # ML regressors & walk-forward CV splitter
│   ├── uncertainty.py                  # Ensemble uncertainty & calibration metrics
│   ├── portfolio.py                    # Convex optimizers (EW, MVO, UAPOML)
│   ├── backtest.py                     # Walk-forward simulation with costs
│   ├── metrics.py                      # Sharpe, Sortino, Drawdown, Calmar, Turnover
│   └── visualization.py                # Publication-quality plotting module
│
├── experiments/
│   ├── run_experiments.py              # Baseline, ablation, and diagnostics
│   └── run_robustness.py               # Transaction costs, lambda, and regimes
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_comparison.ipynb
│   ├── 04_uncertainty_analysis.ipynb
│   └── 05_portfolio_backtest.ipynb
│
├── reports/
│   ├── final_report.md                 # 20-section comprehensive technical paper
│   ├── interview_prep.md               # 40+ quantitative interview Q&A
│   ├── resume_claims.md                # Resume claim evidence verification
│   ├── figures/                        # Generated PNG diagnostic figures
│   └── tables/                         # Output CSV/Markdown performance tables
│
└── tests/
    ├── test_features.py                # Leakage & mathematical unit tests
    ├── test_uncertainty.py             # Uncertainty positivity & shapes
    ├── test_portfolio.py               # Weight constraints & shrinkage tests
    └── test_backtest.py                # Drawdown & metric verification
```

---

## 🚀 Quickstart & Reproduction

### 1. Installation
```bash
git clone https://github.com/your-username/UAPOML.git
cd UAPOML
pip install -r requirements.txt
```

### 2. Run Full Quantitative Pipeline
```bash
python run_pipeline.py
```
*Downloads data, engineers features, trains walk-forward models, quantifies uncertainty, runs portfolio optimizations, computes all metrics, and generates publication figures in `reports/figures/`.*

### 3. Run Unit Tests
```bash
pytest tests/ -v
```

### 4. Launch Interactive Streamlit Dashboard
```bash
streamlit run app.py
```

---

## 📊 Key Empirical Findings

1. **Uncertainty Calibration:** Realized out-of-sample forecast error (MAE and RMSE) monotonically increases across uncertainty quintiles ($Q_1 \to Q_5$), confirming that predictive uncertainty is economically informative.
2. **Superior Risk-Adjusted Returns:** UAPOML achieves higher Sharpe and Sortino ratios than both classical historical MVO and naive ML-MVO after deducting 10 bps transaction costs.
3. **Turnover Reduction:** Penalizing prediction uncertainty significantly dampens erratic rebalancing shifts, reducing total turnover and execution drag.
4. **Stress Regime Capital Preservation:** During the 2020 COVID shock and 2022 rate hike drawdown, UAPOML rotated into defensive, low-uncertainty assets, mitigating portfolio drawdown.

---

## 📚 Technical Interview Resources
- **[Interview Preparation Document](reports/interview_prep.md):** 30s/1m/3m pitches, design decisions, and 40+ deep quantitative Q&As.
- **[Resume Claim Verification](reports/resume_claims.md):** Line-by-line mapping of resume bullets to experimental code and verified empirical evidence.
- **[Full Technical Report](reports/final_report.md):** Comprehensive 20-section academic research paper.

---

## 📄 License
MIT License. Free for academic research and portfolio demonstration.
