import pandas as pd

from abc import ABC, abstractmethod

class DataframePipelineStep(ABC):
    @abstractmethod
    def fit(self, df: pd.DataFrame, **kwargs) -> "DataframePipelineStep":
        return self

    @abstractmethod
    def transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        pass

    def fit_transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return self.fit(df, **kwargs).transform(df, **kwargs)
