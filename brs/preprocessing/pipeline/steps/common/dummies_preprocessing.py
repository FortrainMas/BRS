import pandas as pd
from collections.abc import Sequence
from brs.pipelines.dataframe import DataframePipelineStep


class DummiesPreprocessing(DataframePipelineStep):
    def __init__(self, columns: Sequence[str], drop_first: bool = False):
        self.columns = list(columns)
        self.drop_first = drop_first
        self.new_columns_: list[str] | None = None

    def fit(self, df: pd.DataFrame, **kwargs) -> "DummiesPreprocessing":
        dummies = pd.get_dummies(
            df[self.columns], drop_first=self.drop_first, dtype=int
        )
        self.new_columns_ = dummies.columns.tolist()
        return self

    def transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        if self.new_columns_ is None:
            raise ValueError("DummiesPreprocessing must be fitted first")

        df = df.copy()
        dummies = pd.get_dummies(
            df[self.columns], drop_first=self.drop_first, dtype=int
        )

        for col in self.new_columns_:
            if col not in dummies.columns:
                dummies[col] = 0
        dummies = dummies[self.new_columns_]

        df = df.drop(columns=self.columns)
        df = pd.concat([df, dummies], axis=1)

        return df

    def fit_transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return super().fit_transform(df, **kwargs)
