"""
Bayesian Structural Time Series (BSTS) model using statsmodels.
Returns forecasts and macro coefficient importance.
"""

import numpy as np
import pandas as pd
import warnings
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

warnings.filterwarnings("ignore")

class BSTSPredictor:
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
        
    def fit_predict(self, returns: pd.Series, predictors: pd.DataFrame, use_full_window: bool = False) -> dict:
        """
        Fit BSTS model and return forecast.
        
        Args:
            returns: pd.Series of log returns with DatetimeIndex
            predictors: pd.DataFrame of macro features with DatetimeIndex
            use_full_window: If True, regression fallback uses ALL data (shrinking windows).
                           If False, regression fallback uses last 60 days (daily active).
        """
        # Align data
        aligned = predictors.reindex(returns.index).ffill().dropna()
        common_idx = returns.index.intersection(aligned.index)
        y = returns.loc[common_idx].values
        X = aligned.loc[common_idx].values
        feature_names = aligned.columns.tolist()
        
        if len(y) < 100 or X.shape[1] == 0:
            return self._naive_forecast(y, feature_names)
        
        # Try statsmodels
        try:
            model = sm.tsa.UnobservedComponents(
                y,
                level='local level',
                trend=True,
                exog=X,
                autoregressive=0
            )
            fit = model.fit(disp=False, maxiter=500)
            forecast = fit.get_forecast(steps=1, exog=X[-1:].reshape(1, -1))
            pred_mean = forecast.predicted_mean[0]
            ci = forecast.conf_int(alpha=0.05)
            pred_lower = ci[0, 0]
            pred_upper = ci[0, 1]
            
            # Extract exog coefficients
            coeffs = None
            if hasattr(fit, 'params'):
                n_exog = X.shape[1]
                if len(fit.params) >= 3 + n_exog:
                    coeffs = fit.params[3:3+n_exog]
            
            if coeffs is None:
                return self._regression_forecast(y, X, feature_names, use_full_window)
            
            importance = self._compute_importance(coeffs, feature_names)
            return {
                'forecast_mean': pred_mean,
                'forecast_lower': pred_lower,
                'forecast_upper': pred_upper,
                'macro_importance': importance
            }
        except Exception as e:
            print(f"Statsmodels BSTS failed: {e}. Using regression fallback.")
            return self._regression_forecast(y, X, feature_names, use_full_window)
    
    def _regression_forecast(self, y, X, feature_names, use_full_window: bool = False):
        """
        Fallback linear regression forecast.
        
        Args:
            use_full_window: If True, uses ALL data (shrinking windows).
                           If False, uses last 60 days (daily active).
        """
        if use_full_window:
            # Shrinking windows: use ALL data for that window
            window = len(y) - 1
        else:
            # Daily active: use last 60 days (recency bias)
            window = min(60, len(y) - 1)
            
        if window < 20:
            return self._naive_forecast(y, feature_names)
        
        y_train = y[-window:]
        X_train = X[-window:]
        X_next = X[-1:].reshape(1, -1)
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        pred = model.predict(X_next)[0]
        resid_std = np.std(y_train - model.predict(X_train))
        
        coeffs = model.coef_
        importance = self._compute_importance(coeffs, feature_names)
        
        return {
            'forecast_mean': pred,
            'forecast_lower': pred - 1.96 * resid_std,
            'forecast_upper': pred + 1.96 * resid_std,
            'macro_importance': importance
        }
    
    def _compute_importance(self, coefficients, feature_names):
        if coefficients is None or len(coefficients) != len(feature_names):
            return []
        imp = np.abs(coefficients)
        sorted_idx = np.argsort(imp)[::-1]
        result = []
        for i in sorted_idx:
            if imp[i] > 1e-8:
                result.append({
                    'feature': feature_names[i],
                    'coefficient': float(coefficients[i]),
                    'importance': float(imp[i])
                })
        return result
    
    def _naive_forecast(self, y, feature_names):
        recent = y[-21:] if len(y) >= 21 else y
        mean = np.mean(recent)
        std = np.std(recent)
        return {
            'forecast_mean': mean,
            'forecast_lower': mean - 1.96 * std,
            'forecast_upper': mean + 1.96 * std,
            'macro_importance': []
        }
