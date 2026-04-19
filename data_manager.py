"""
Data loading and preprocessing for BSTS engine.
"""

import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
import config

def load_master_data() -> pd.DataFrame:
    """
    Downloads master_data.parquet from Hugging Face and loads into DataFrame.
    Handles DatetimeIndex and wide-format price data.
    """
    print(f"Downloading {config.HF_DATA_FILE} from {config.HF_DATA_REPO}...")
    file_path = hf_hub_download(
        repo_id=config.HF_DATA_REPO,
        filename=config.HF_DATA_FILE,
        repo_type="dataset",
        token=config.HF_TOKEN,
        cache_dir="./hf_cache"
    )
    df = pd.read_parquet(file_path)
    print(f"Loaded {len(df)} rows from master data.")
    
    # Ensure Date is a column
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index().rename(columns={'index': 'Date'})
    elif 'Date' not in df.columns and 'date' in df.columns:
        df = df.rename(columns={'date': 'Date'})
    df['Date'] = pd.to_datetime(df['Date'])
    
    return df

def compute_log_returns(df: pd.DataFrame, tickers: list) -> pd.DataFrame:
    """
    Compute log returns for given tickers from price columns.
    Returns a long-format DataFrame with Date, ticker, log_return.
    """
    # Keep only ticker columns that exist in the data
    available_tickers = [t for t in tickers if t in df.columns]
    print(f"Found {len(available_tickers)} ticker columns out of {len(tickers)} expected.")
    
    # Melt to long format
    df_long = pd.melt(
        df,
        id_vars=['Date'],
        value_vars=available_tickers,
        var_name='ticker',
        value_name='price'
    )
    
    # Compute log returns per ticker
    df_long = df_long.sort_values(['ticker', 'Date'])
    df_long['log_return'] = df_long.groupby('ticker')['price'].transform(
        lambda x: np.log(x / x.shift(1))
    )
    df_long = df_long.dropna(subset=['log_return'])
    
    return df_long[['Date', 'ticker', 'log_return']]

def prepare_macro_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract macro columns and compute daily changes for predictors.
    Returns DataFrame with Date index and macro features.
    """
    macro_df = df[['Date'] + [c for c in config.MACRO_COLS if c in df.columns]].copy()
    macro_df = macro_df.set_index('Date')
    
    # Use raw values (or optionally compute changes)
    # For BSTS regression, we'll use the raw macro levels
    return macro_df.ffill().dropna()  # Fixed FutureWarning
