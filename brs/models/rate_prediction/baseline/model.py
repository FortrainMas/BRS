from pathlib import Path
from typing import Any
import pandas as pd

from brs.models.model import BaseModel

class BaselineRateModel(BaseModel):
    def __init__(self):
        super().__init__()
    
    def fit(self, X:pd.DataFrame, y=None) -> "BaselineRateModel":
        return self
    
    def predict(self, X: pd.DataFrame) -> pd.Series:
        return X["Mean-Book-Rating"]
    
    def save(self, path: Path):
        pass

    def load(self, path: Path) -> "BaselineRateModel":
        return self