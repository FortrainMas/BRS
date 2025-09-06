from pathlib import Path
import joblib

import pandas as pd
from sklearn.linear_model import LinearRegression

from brs.models.model import BaseModel

class LinRegModel(BaseModel):
    def __init__(self):
        self.model = LinearRegression()
    
    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "LinRegModel":
        if y is None:
            raise ValueError("y should not be None")

        self.model.fit(X, y)
        return self
    
    def predict(self, X: pd.DataFrame) -> pd.Series:
        preds = self.model.predict(X)
        return pd.Series(preds, index=X.index, name="Book-Rating")
    
    def save(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
    
    def load(self, path: Path) -> "LinRegModel":
        path = Path(path)
        self.model = joblib.load(path)
        return self
