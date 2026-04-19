"""
Bayesian Structural Time Series (BSTS) model using pybuc.
"""

import numpy as np
import pandas as pd
from pybuc import buc

class BSTSPredictor:
    """
    BSTS model with level, trend, and spike‑and‑slab regression on macro predictors.
    """
    
    def __init__(self, mcmc_samples: int = 2000, mcmc_burn: int = 500, seed: int = 42):
        self.mcmc_samples = mcmc_samples
        self.mcmc_burn = mcmc_burn
        self.seed = seed
        buc.set_seed(seed)
        
    def fit_predict(self, returns: pd.Series, predictors: pd.DataFrame) -> dict:
        """
        Fit BSTS model on historical returns and predictors, then forecast next day.
        Returns dictionary with forecast mean, lower/upper bounds, and model summary.
        """
        # Align predictors with returns index
        aligned_predictors = predictors.reindex(returns.index).fillna(method='ffill').dropna()
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
        
        try:
            # Build BSTS model with level, trend, and regression component
            model = buc.BayesianUnobservedComponents(
                response=y,
                level=True,
                stochastic_level=True,
                slope=True,
                stochastic_slope=True,
                predictors=X,
                # Use default priors (inverse-Gamma for variances, Gaussian for coefficients)
            )
            
            # Sample from posterior
            model.sample(self.mcmc_samples)
            
            # Forecast next step
            forecast_samples = model.forecast(steps=1, burn=self.mcmc_burn)
            forecast_mean = np.mean(forecast_samples)
            forecast_lower = np.percentile(forecast_samples, 2.5)
            forecast_upper = np.percentile(forecast_samples, 97.5)
            
            # Extract model summary (optional)
            summary = model.summary(burn=self.mcmc_burn)
            
            return {
                'forecast_mean': forecast_mean,
                'forecast_lower': forecast_lower,
                'forecast_upper': forecast_upper,
                'model_summary': summary
            }
        except Exception as e:
            print(f"BSTS model fitting failed: {e}")
            return {
                'forecast_mean': np.nan,
                'forecast_lower': np.nan,
                'forecast_upper': np.nan,
                'error': str(e)
            }
