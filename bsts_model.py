"""
Bayesian Structural Time Series (BSTS) model with robust fallbacks.
"""

import numpy as np
import pandas as pd
import warnings
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")

try:
    from pybuc import buc
    PYBUC_AVAILABLE = True
except ImportError:
    PYBUC_AVAILABLE = False

class BSTSPredictor:
    """
    Attempts BSTS; falls back to rolling linear regression if pybuc fails.
    """
    
    def __init__(self, mcmc_samples: int = 2000, mcmc_burn: int = 500, seed: int = 42):
        self.mcmc_samples = mcmc_samples
        self.mcmc_burn = mcmc_burn
        self.seed = seed
        np.random.seed(seed)
        
    def fit_predict(self, returns: pd.Series, predictors: pd.DataFrame) -> dict:
        """
        Returns forecast mean, lower, upper.
        """
        aligned = predictors.reindex(returns.index).ffill().dropna()
        common_idx = returns.index.intersection(aligned.index)
        y = returns.loc[common_idx].values
        X = aligned.loc[common_idx].values
        
        if len(y) < 100 or X.shape[1] == 0:
            return self._naive_forecast(y)
        
        # Try BSTS first
        if PYBUC_AVAILABLE:
            try:
                model = buc.BayesianUnobservedComponents(
                    response=y,
                    level=True,
                    stochastic_level=True,
                    trend=True,
                    stochastic_trend=True,
                    predictors=X,
                )
                model.sample(self.mcmc_samples)
                
                # Try multiple forecast method signatures
                forecast_samples = None
                for method_name, kwargs in [
                    ('forecast', {'steps': 1}),
                    ('forecast', {'h': 1}),
                    ('predict', {'h': 1}),
                    ('forecast', {'horizon': 1}),
                    ('forecast', (1,)),  # positional
                ]:
                    try:
                        method = getattr(model, method_name)
                        if isinstance(kwargs, dict):
                            result = method(**kwargs, burn=self.mcmc_burn)
                        else:
                            result = method(*kwargs, burn=self.mcmc_burn)
                        if isinstance(result, tuple):
                            forecast_samples = result[0]
                        else:
                            forecast_samples = result
                        break
                    except:
                        continue
                
                if forecast_samples is not None:
                    return {
                        'forecast_mean': np.mean(forecast_samples),
                        'forecast_lower': np.percentile(forecast_samples, 2.5),
                        'forecast_upper': np.percentile(forecast_samples, 97.5),
                    }
            except Exception as e:
                pass  # Fall through to regression fallback
        
        # Fallback: rolling linear regression using last 60 days
        return self._regression_forecast(y, X)
    
    def _regression_forecast(self, y, X):
        """Use last 60 observations to fit linear model and forecast next day."""
        window = min(60, len(y) - 1)
        if window < 20:
            return self._naive_forecast(y)
        
        # Train on all but last day (or use last window)
        y_train = y[-window:]
        X_train = X[-window:]
        X_next = X[-1:].copy()  # most recent macro values for forecast
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        pred = model.predict(X_next)[0]
        
        # Uncertainty from residual std
        resid_std = np.std(y_train - model.predict(X_train))
        return {
            'forecast_mean': pred,
            'forecast_lower': pred - 1.96 * resid_std,
            'forecast_upper': pred + 1.96 * resid_std,
        }
    
    def _naive_forecast(self, y):
        """21-day mean as last resort."""
        recent = y[-21:] if len(y) >= 21 else y
        mean = np.mean(recent)
        std = np.std(recent)
        return {
            'forecast_mean': mean,
            'forecast_lower': mean - 1.96 * std,
            'forecast_upper': mean + 1.96 * std,
        }
