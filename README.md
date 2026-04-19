# P2-ETF-BSTS-TREND

**Bayesian Structural Time Series with Spike‑and‑Slab Regression for ETF Forecasting**

[![Daily Run](https://github.com/P2SAMAPA/P2-ETF-BSTS-TREND/actions/workflows/daily_run.yml/badge.svg)](https://github.com/P2SAMAPA/P2-ETF-BSTS-TREND/actions/workflows/daily_run.yml)
[![Hugging Face Dataset](https://img.shields.io/badge/🤗%20Dataset-p2--etf--bsts--trend--results-blue)](https://huggingface.co/datasets/P2SAMAPA/p2-etf-bsts-trend-results)

## Overview

`P2-ETF-BSTS-TREND` is a quantitative forecasting engine that predicts next‑day ETF returns using **Bayesian Structural Time Series (BSTS)** models. Each ETF is modeled with:

- A **stochastic level and trend** component (captures underlying drift)
- A **spike‑and‑slab regression** component on macro predictors (VIX, DXY, spreads, etc.)
- **Gibbs sampling** for full posterior inference

The engine outputs daily forecasts for 23 ETFs across three universes and identifies the **top pick** (highest expected return) for each universe.

## Universe Coverage

| Universe | Tickers |
|----------|---------|
| **FI / Commodities** | TLT, VCIT, LQD, HYG, VNQ, GLD, SLV |
| **Equity Sectors** | SPY, QQQ, XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU, GDX, XME, IWF, XSD, XBI, IWM |
| **Combined** | All tickers above |

## Methodology

### Bayesian Structural Time Series (BSTS)

The model decomposes each ETF's log‑return into:

y_t = μ_t + δ_t + x_t'β + ε_t

where:
- **μ_t**: stochastic level (random walk)
- **δ_t**: stochastic trend (slope)
- **x_t'β**: regression on macro predictors (VIX, DXY, spreads, etc.)
- **ε_t**: observation noise

### Spike‑and‑Slab Prior

The regression coefficients β use a **spike‑and‑slab prior**—a mixture of a point mass at zero ("spike") and a wide Gaussian ("slab"). This promotes **coefficient sparsity** and automatic variable selection, preventing overfitting when many predictors are included.

### Inference

Posterior samples are drawn via **Gibbs sampling** (2,000 draws with 500 burn‑in). The forecast for the next day is the posterior mean, with 95% credible intervals.

## File Structure
P2-ETF-BSTS-TREND/
├── config.py # Paths, universes, BSTS parameters
├── data_manager.py # Data loading and preprocessing
├── bsts_model.py # Core BSTS forecasting logic
├── trainer.py # Main orchestration script
├── push_results.py # Upload results to Hugging Face
├── streamlit_app.py # Interactive dashboard
├── requirements.txt # Python dependencies
├── .github/workflows/ # Scheduled GitHub Action
└── .streamlit/ # Streamlit theme

text

## Running Locally

```bash
git clone https://github.com/P2SAMAPA/P2-ETF-BSTS-TREND.git
cd P2-ETF-BSTS-TREND
pip install -r requirements.txt
export HF_TOKEN="your_token_here"
python trainer.py
streamlit run streamlit_app.py
Dashboard Features
Hero Card: Displays the top ETF pick with forecasted return and 95% credible interval

Three Tabs: Separate views for Combined, Equity, and FI/Commodities universes

Forecast Table: Sortable list of all ETF predictions

Bar Chart: Visual ranking of forecasted returns

License
MIT License

text

---

### What’s Different from Previous Engines

| Feature | This Engine |
|---------|-------------|
| **Forecast Target** | Next‑day return (direction + magnitude) |
| **Methodology** | Bayesian structural time series with spike‑and‑slab regression |
| **Output** | Top ETF pick per universe displayed in a hero card |
| **Uncertainty** | 95% credible intervals from posterior sampling |
