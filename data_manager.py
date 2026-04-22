"""
Data loading and preprocessing for BSTS engine.
"""

import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
import config

def load_master_data() -> pd.DataFrame:
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
    
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index().rename(columns={'index': 'Date'})
    elif 'Date' not in df.columns and 'date' in df.columns:
        df = df.rename(columns={'date': 'Date'})
    df['Date'] = pd.to_datetime(df['Date'])
    
    df = df.sort_values('Date').reset_index(drop=True)
    
    return df

def compute_log_returns(df: pd.DataFrame, tickers: list) -> pd.DataFrame:
    available_tickers = [t for t in tickers if t in df.columns]
    print(f"Found {len(available_tickers)} ticker columns out of {len(tickers)} expected.")
    
    df_long = pd.melt(
        df,
        id_vars=['Date'],
        value_vars=available_tickers,
        var_name='ticker',
        value_name='price'
    )
    
    df_long = df_long.sort_values(['ticker', 'Date'])
    df_long['log_return'] = df_long.groupby('ticker')['price'].transform(
        lambda x: np.log(x / x.shift(1))
    )
    df_long = df_long.dropna(subset=['log_return'])
    
    return df_long[['Date', 'ticker', 'log_return']]

def prepare_macro_features(df: pd.DataFrame) -> pd.DataFrame:
    macro_df = df[['Date'] + [c for c in config.MACRO_COLS if c in df.columns]].copy()
    macro_df = macro_df.set_index('Date').sort_index()
    return macro_df.ffill().dropna()
