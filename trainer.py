"""
Main training script for BSTS Trend engine.
Fits BSTS models per ETF, forecasts next‑day returns, and pushes results to HF.
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
    
    all_results = {}
    top_picks = {}
    
    predictor = BSTSPredictor(
        mcmc_samples=config.MCMC_SAMPLES,
        mcmc_burn=config.MCMC_BURN,
        seed=config.RANDOM_SEED
    )
    
    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n--- Processing Universe: {universe_name} ---")
        universe_results = {}
        
        # Compute returns for this universe
        returns_df = data_manager.compute_log_returns(df_master, tickers)
        
        for ticker in tickers:
            print(f"  Forecasting {ticker}...")
            ticker_returns = returns_df[returns_df['ticker'] == ticker].set_index('Date')['log_return']
            
            if len(ticker_returns) < config.LOOKBACK_WINDOW:
                print(f"    Skipping {ticker}: insufficient data ({len(ticker_returns)} obs)")
                continue
            
            # Use recent window
            ticker_returns = ticker_returns.iloc[-config.LOOKBACK_WINDOW:]
            
            # Fit and forecast
            forecast = predictor.fit_predict(ticker_returns, macro_features)
            
            universe_results[ticker] = {
                'ticker': ticker,
                'forecast_mean': forecast.get('forecast_mean'),
                'forecast_lower': forecast.get('forecast_lower'),
                'forecast_upper': forecast.get('forecast_upper'),
                'error': forecast.get('error')
            }
        
        # Identify top pick for this universe (highest forecast mean)
        valid_forecasts = {t: d for t, d in universe_results.items() if d['forecast_mean'] is not None and not np.isnan(d['forecast_mean'])}
        if valid_forecasts:
            top_ticker = max(valid_forecasts, key=lambda t: valid_forecasts[t]['forecast_mean'])
            top_picks[universe_name] = {
                'ticker': top_ticker,
                'forecast_mean': valid_forecasts[top_ticker]['forecast_mean'],
                'forecast_lower': valid_forecasts[top_ticker]['forecast_lower'],
                'forecast_upper': valid_forecasts[top_ticker]['forecast_upper']
            }
        
        all_results[universe_name] = universe_results
    
    # Build output payload
    output_payload = {
        "run_date": config.TODAY,
        "config": {
            "lookback_window": config.LOOKBACK_WINDOW,
            "mcmc_samples": config.MCMC_SAMPLES,
            "mcmc_burn": config.MCMC_BURN,
            "random_seed": config.RANDOM_SEED
        },
        "top_picks": top_picks,
        "universes": all_results
    }
    
    # Push to Hugging Face
    push_results.push_daily_result(output_payload)
    
    # Print top picks
    print("\n=== Top Picks for Next Day ===")
    for universe, pick in top_picks.items():
        print(f"{universe}: {pick['ticker']} with forecast return {pick['forecast_mean']*100:.3f}% "
              f"({pick['forecast_lower']*100:.3f}% to {pick['forecast_upper']*100:.3f}%)")
    
    print("\n=== Run Complete ===")

if __name__ == "__main__":
    run_bsts_forecast()
