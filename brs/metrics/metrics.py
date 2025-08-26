from abc import ABC, abstractmethod

import pandas as pd


class MetricsStrategy(ABC):
    @abstractmethod
    def estimate(self, y_pred: pd.Series, y_true: pd.Series) -> float:
        pass
