"""
Bayesian Structural Time Series (BSTS) model using statsmodels.
"""

import numpy as np
import pandas as pd
import warnings
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

warnings.filterwarnings("ignore")

class BSTSPredictor:
    """
    Uses statsmodels UnobservedComponents for BSTS with level and trend.
    Falls back to linear regression on macro predictors if that fails.
    """
    
    def __init__(self, mcmc_samples: int = 2000, mcmc_burn: int = 500, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
        
    def fit_predict(self, returns: pd.Series, predictors: pd.DataFrame) -> dict:
        # Align data
        aligned = predictors.reindex(returns.index).ffill().dropna()
        common_idx = returns.index.intersection(aligned.index)
        y = returns.loc[common_idx].values
        X = aligned.loc[common_idx].values
        
        if len(y) < 100:
            return self._naive_forecast(y)
        
        # Try statsmodels UnobservedComponents (BSTS equivalent)
        try:
            # Build model with local level + deterministic trend
            model = sm.tsa.UnobservedComponents(
                y,
                level='local level',
                trend=True,
                exog=X if X.shape[1] > 0 else None,
                autoregressive=0
            )
            fit = model.fit(disp=False, maxiter=100)
            forecast = fit.get_forecast(steps=1, exog=X[-1:].reshape(1, -1) if X.shape[1] > 0 else None)
            pred_mean = forecast.predicted_mean[0]
            ci = forecast.conf_int(alpha=0.05)
            pred_lower = ci[0, 0]
            pred_upper = ci[0, 1]
            
            return {
                'forecast_mean': pred_mean,
                'forecast_lower': pred_lower,
                'forecast_upper': pred_upper,
            }
        except Exception as e:
            print(f"Statsmodels BSTS failed: {e}. Using regression fallback.")
            return self._regression_forecast(y, X)
    
    def _regression_forecast(self, y, X):
        """Use last 60 days to fit linear model on macro predictors."""
        window = min(60, len(y) - 1)
        if window < 20 or X.shape[1] == 0:
            return self._naive_forecast(y)
        
        y_train = y[-window:]
        X_train = X[-window:]
        X_next = X[-1:].reshape(1, -1)
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        pred = model.predict(X_next)[0]
        resid_std = np.std(y_train - model.predict(X_train))
        
        return {
            'forecast_mean': pred,
            'forecast_lower': pred - 1.96 * resid_std,
            'forecast_upper': pred + 1.96 * resid_std,
        }
    
    def _naive_forecast(self, y):
        """21-day mean fallback."""
        recent = y[-21:] if len(y) >= 21 else y
        mean = np.mean(recent)
        std = np.std(recent)
        return {
            'forecast_mean': mean,
            'forecast_lower': mean - 1.96 * std,
            'forecast_upper': mean + 1.96 * std,
        }
