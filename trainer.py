"""
Main training script for BSTS Trend engine.
Computes both daily (rolling 504d) and shrinking-window forecasts.
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime

import config
import data_manager
from bsts_model import BSTSPredictor
import push_results

def run_bsts_forecast():
    """Orchestrates the full BSTS forecasting pipeline."""
    
    print(f"=== P2-ETF-BSTS-TREND Run: {config.TODAY} ===")
    
    # Load and prepare data
    df_master = data_manager.load_master_data()
    macro_features = data_manager.prepare_macro_features(df_master)
    
    predictor = BSTSPredictor(seed=config.RANDOM_SEED)
    
    all_results = {}
    top_picks = {}
    
    # 1. Daily Active Trading (rolling 504d window)
    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n--- Daily Active: {universe_name} ---")
        universe_results = {}
        returns_df = data_manager.compute_log_returns(df_master, tickers)
        
        for ticker in tickers:
            print(f"  Forecasting {ticker}...")
            ticker_returns = returns_df[returns_df['ticker'] == ticker].set_index('Date')['log_return']
            if len(ticker_returns) < config.LOOKBACK_WINDOW:
                continue
            ticker_returns = ticker_returns.iloc[-config.LOOKBACK_WINDOW:]
            forecast = predictor.fit_predict(ticker_returns, macro_features)
            universe_results[ticker] = {
                'ticker': ticker,
                'forecast_mean': forecast.get('forecast_mean'),
                'forecast_lower': forecast.get('forecast_lower'),
                'forecast_upper': forecast.get('forecast_upper'),
            }
        
        valid = {t: d for t, d in universe_results.items() if d['forecast_mean'] is not None and not np.isnan(d['forecast_mean'])}
        if valid:
            top_ticker = max(valid, key=lambda t: valid[t]['forecast_mean'])
            top_picks[universe_name] = {
                'ticker': top_ticker,
                'forecast_mean': valid[top_ticker]['forecast_mean'],
                'forecast_lower': valid[top_ticker]['forecast_lower'],
                'forecast_upper': valid[top_ticker]['forecast_upper']
            }
        all_results[universe_name] = universe_results
    
    # 2. Shrinking Windows (full history down to 2 years)
    shrinking_results = {}
    for start_year in config.SHRINKING_WINDOW_START_YEARS:
        start_date = pd.Timestamp(f"{start_year}-01-01")
        window_label = f"{start_year}-{config.TODAY[:4]}"
        print(f"\n--- Shrinking Window: {window_label} ---")
        
        # Filter data from start_date onward
        mask = df_master['Date'] >= start_date
        df_window = df_master[mask].copy()
        if len(df_window) < 252:  # need at least 1 year
            continue
        
        macro_win = macro_features.loc[start_date:].dropna()
        window_results = {}
        
        for universe_name, tickers in config.UNIVERSES.items():
            universe_returns = data_manager.compute_log_returns(df_window, tickers)
            for ticker in tickers:
                ticker_ret = universe_returns[universe_returns['ticker'] == ticker].set_index('Date')['log_return']
                if len(ticker_ret) < 252:
                    continue
                forecast = predictor.fit_predict(ticker_ret, macro_win)
                window_results.setdefault(universe_name, {})[ticker] = {
                    'forecast_mean': forecast.get('forecast_mean'),
                    'forecast_lower': forecast.get('forecast_lower'),
                    'forecast_upper': forecast.get('forecast_upper'),
                }
        
        # Determine top pick per universe for this window
        window_top = {}
        for universe_name, ticker_dict in window_results.items():
            valid = {t: d for t, d in ticker_dict.items() if d['forecast_mean'] is not None and not np.isnan(d['forecast_mean'])}
            if valid:
                top_ticker = max(valid, key=lambda t: valid[t]['forecast_mean'])
                window_top[universe_name] = {
                    'ticker': top_ticker,
                    'forecast_mean': valid[top_ticker]['forecast_mean'],
                    'forecast_lower': valid[top_ticker]['forecast_lower'],
                    'forecast_upper': valid[top_ticker]['forecast_upper']
                }
        shrinking_results[window_label] = {
            'start_year': start_year,
            'start_date': start_date.isoformat(),
            'forecasts': window_results,
            'top_picks': window_top
        }
    
    # Build output payload
    output_payload = {
        "run_date": config.TODAY,
        "config": {
            "lookback_window": config.LOOKBACK_WINDOW,
            "shrinking_start_years": config.SHRINKING_WINDOW_START_YEARS
        },
        "daily_active": {
            "top_picks": top_picks,
            "universes": all_results
        },
        "shrinking_windows": shrinking_results
    }
    
    # Push to Hugging Face
    push_results.push_daily_result(output_payload)
    
    # Print top picks
    print("\n=== Daily Active Top Picks ===")
    for universe, pick in top_picks.items():
        print(f"{universe}: {pick['ticker']} with forecast return {pick['forecast_mean']*100:.3f}%")
    
    print("\n=== Run Complete ===")

if __name__ == "__main__":
    run_bsts_forecast()
