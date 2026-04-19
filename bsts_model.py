"""
Bayesian Structural Time Series (BSTS) model using statsmodels.
Returns forecasts and macro coefficient importance.
"""

import numpy as np
import pandas as pd
import warnings
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

warnings.filterwarnings("ignore")

class BSTSPredictor:
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
        
    def fit_predict(self, returns: pd.Series, predictors: pd.DataFrame) -> dict:
        # Align data
        aligned = predictors.reindex(returns.index).ffill().dropna()
        common_idx = returns.index.intersection(aligned.index)
        y = returns.loc[common_idx].values
        X = aligned.loc[common_idx].values
        feature_names = aligned.columns.tolist()
        
        if len(y) < 100:
            return self._naive_forecast(y)
        
        # Try statsmodels UnobservedComponents
        try:
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
            
            # Extract coefficients (if exog used)
            coeffs = None
            if X.shape[1] > 0 and hasattr(fit, 'params'):
                # params order: [sigma2.irregular, sigma2.level, sigma2.trend, exog_coeffs...]
                exog_start = 3  # after variances
                if len(fit.params) >= exog_start + X.shape[1]:
                    coeffs = fit.params[exog_start:exog_start + X.shape[1]]
            
            return {
                'forecast_mean': pred_mean,
                'forecast_lower': pred_lower,
                'forecast_upper': pred_upper,
                'macro_importance': self._compute_importance(coeffs, feature_names) if coeffs is not None else None
            }
        except Exception as e:
            print(f"Statsmodels BSTS failed: {e}. Using regression fallback.")
            return self._regression_forecast(y, X, feature_names)
    
    def _regression_forecast(self, y, X, feature_names):
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
        
        coeffs = model.coef_
        importance = self._compute_importance(coeffs, feature_names)
        
        return {
            'forecast_mean': pred,
            'forecast_lower': pred - 1.96 * resid_std,
            'forecast_upper': pred + 1.96 * resid_std,
            'macro_importance': importance
        }
    
    def _compute_importance(self, coefficients, feature_names):
        """Convert raw coefficients to absolute importance scores (sorted)."""
        if coefficients is None or len(coefficients) != len(feature_names):
            return None
        # Absolute value of coefficients as importance
        imp = np.abs(coefficients)
        # Sort descending
        sorted_idx = np.argsort(imp)[::-1]
        return [
            {'feature': feature_names[i], 'coefficient': float(coefficients[i]), 'importance': float(imp[i])}
            for i in sorted_idx if imp[i] > 1e-6
        ]
    
    def _naive_forecast(self, y):
        recent = y[-21:] if len(y) >= 21 else y
        mean = np.mean(recent)
        std = np.std(recent)
        return {
            'forecast_mean': mean,
            'forecast_lower': mean - 1.96 * std,
            'forecast_upper': mean + 1.96 * std,
            'macro_importance': None
        }
