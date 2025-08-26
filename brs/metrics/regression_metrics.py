import pandas as pd
import numpy as np

from .metrics import MetricsStrategy

class MAE(MetricsStrategy):
    def estimate(self, y_pred: pd.Series, y_true: pd.Series) -> float:
        return float(np.mean(np.abs(y_true - y_pred)))
    
class MSE(MetricsStrategy):
    def estimate(self, y_pred: pd.Series, y_true: pd.Series) -> float:
        return float(np.mean(np.square(y_true - y_pred)))
    
class RMSE(MetricsStrategy):
    def estimate(self, y_pred: pd.Series, y_true: pd.Series) -> float:
        return float(np.sqrt(np.mean(np.square(y_true - y_pred))))
