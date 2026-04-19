"""
Configuration for P2-ETF-BSTS-TREND engine.
"""

import os
from datetime import datetime

# --- Hugging Face Repositories ---
HF_DATA_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"
HF_DATA_FILE = "master_data.parquet"

HF_OUTPUT_REPO = "P2SAMAPA/p2-etf-bsts-trend-results"

# --- Universe Definitions (mirroring master data) ---
FI_COMMODITIES_TICKERS = ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"]
EQUITY_SECTORS_TICKERS = [
    "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV",
    "XLI", "XLY", "XLP", "XLU", "GDX", "XME",
    "IWF", "XSD", "XBI", "IWM"
]
ALL_TICKERS = list(set(FI_COMMODITIES_TICKERS + EQUITY_SECTORS_TICKERS))

UNIVERSES = {
    "FI_COMMODITIES": FI_COMMODITIES_TICKERS,
    "EQUITY_SECTORS": EQUITY_SECTORS_TICKERS,
    "COMBINED": ALL_TICKERS
}

# --- Macro Columns (used as predictors) ---
MACRO_COLS = ["VIX", "DXY", "T10Y2Y", "TBILL_3M", "IG_SPREAD", "HY_SPREAD"]

# --- BSTS Parameters ---
LOOKBACK_WINDOW = 504  # 2 years of daily data for model fitting (~504 trading days)
FORECAST_HORIZON = 1   # Predict next day's return
MCMC_SAMPLES = 2000    # Number of Gibbs sampling draws
MCMC_BURN = 500        # Burn-in period for MCMC
RANDOM_SEED = 42       # Reproducibility

# --- Date Handling ---
TODAY = datetime.now().strftime("%Y-%m-%d")

# --- Optional: Hugging Face Token ---
HF_TOKEN = os.environ.get("HF_TOKEN", None)
