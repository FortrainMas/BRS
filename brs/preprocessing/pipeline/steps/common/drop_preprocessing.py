from typing import Sequence

import pandas as pd

from brs.pipelines.dataframe import DataframePipelineStep


class DropPreprocessing(DataframePipelineStep):
    def __init__(self, columns: Sequence[str]):
        self.columns = columns

    def fit(self, df: pd.DataFrame, **kwargs) -> "DropPreprocessing":
        return self

    def transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = df.copy()
        df = df.drop(columns=self.columns, errors="ignore")
        return df

    def fit_transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return super().fit_transform(df, **kwargs)
