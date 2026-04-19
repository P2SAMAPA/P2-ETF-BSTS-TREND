"""
Bayesian Structural Time Series (BSTS) model using pybuc.
"""

import numpy as np
import pandas as pd

try:
    from pybuc import buc
    PYBUC_AVAILABLE = True
except ImportError:
    PYBUC_AVAILABLE = False
    print("Warning: pybuc not installed. Using fallback simple forecast.")

class BSTSPredictor:
    """
    BSTS model with level, trend, and spike‑and‑slab regression on macro predictors.
    """
    
    def __init__(self, mcmc_samples: int = 2000, mcmc_burn: int = 500, seed: int = 42):
        self.mcmc_samples = mcmc_samples
        self.mcmc_burn = mcmc_burn
        self.seed = seed
        np.random.seed(seed)
        
    def fit_predict(self, returns: pd.Series, predictors: pd.DataFrame) -> dict:
        """
        Fit BSTS model on historical returns and predictors, then forecast next day.
        Returns dictionary with forecast mean, lower/upper bounds.
        """
        # Align predictors with returns index
        aligned_predictors = predictors.reindex(returns.index).ffill().dropna()
        common_idx = returns.index.intersection(aligned_predictors.index)
        
        y = returns.loc[common_idx].values
        X = aligned_predictors.loc[common_idx].values
        
        if len(y) < 100 or X.shape[1] == 0:
            return {
                'forecast_mean': np.nan,
                'forecast_lower': np.nan,
                'forecast_upper': np.nan,
                'error': 'Insufficient data'
            }
        
        if not PYBUC_AVAILABLE:
            recent_mean = np.mean(y[-21:])
            recent_std = np.std(y[-21:])
            return {
                'forecast_mean': recent_mean,
                'forecast_lower': recent_mean - 1.96 * recent_std,
                'forecast_upper': recent_mean + 1.96 * recent_std,
                'error': 'pybuc not available, using naive forecast'
            }
        
        try:
            # Build BSTS model
            model = buc.BayesianUnobservedComponents(
                response=y,
                level=True,
                stochastic_level=True,
                trend=True,
                stochastic_trend=True,
                predictors=X,
            )
            
            # Sample from posterior
            model.sample(self.mcmc_samples)
            
            # Forecast next step — use h=1 for horizon
            forecast_result = model.forecast(h=1, burn=self.mcmc_burn)
            
            # forecast_result may be a tuple (mean, intervals) or array of draws
            if isinstance(forecast_result, tuple):
                forecast_samples = forecast_result[0]  # Usually the draws
            else:
                forecast_samples = forecast_result
                
            forecast_mean = np.mean(forecast_samples)
            forecast_lower = np.percentile(forecast_samples, 2.5)
            forecast_upper = np.percentile(forecast_samples, 97.5)
            
            return {
                'forecast_mean': forecast_mean,
                'forecast_lower': forecast_lower,
                'forecast_upper': forecast_upper,
            }
        except Exception as e:
            print(f"BSTS model fitting failed: {e}")
            recent_mean = np.mean(y[-21:])
            recent_std = np.std(y[-21:])
            return {
                'forecast_mean': recent_mean,
                'forecast_lower': recent_mean - 1.96 * recent_std,
                'forecast_upper': recent_mean + 1.96 * recent_std,
                'error': str(e)
            }
