import pandas as pd
from collections.abc import Sequence
from brs.pipelines.dataframe import DataframePipelineStep


class TopKDummiesPreprocessing(DataframePipelineStep):
    def __init__(self, columns: Sequence[str], top_k: int = 10, drop_first: bool = False):
        self.columns = list(columns)
        self.top_k = top_k
        self.drop_first = drop_first
        self.top_categories_: dict[str, list[str]] = {}

    def fit(self, df: pd.DataFrame, **kwargs) -> "TopKDummiesPreprocessing":
        for col in self.columns:
            counts = df[col].value_counts()
            self.top_categories_[col] = counts.head(self.top_k).index.tolist()
        return self

    def transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = df.copy()
        for col in self.columns:
            top = self.top_categories_[col]
            df[col] = df[col].where(df[col].isin(top), other="Other")
            dummies = pd.get_dummies(df[col], drop_first=self.drop_first, dtype=int)
            df = df.drop(columns=[col])
            df = pd.concat([df, dummies], axis=1)
        return df

    def fit_transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return self.fit(df, **kwargs).transform(df, **kwargs)
