import pandas as pd
import numpy as np
from brs.pipelines.dataframe import DataframePipelineStep


class NaPreprocessing(DataframePipelineStep):
    def __init__(self, columns: dict[str, dict[str, object]]):
        """
        columns: dict exacmple:
        {
            "Book-Rating": {"strategy": "mean"},
            "Age": {"strategy": "constant", "const_value": 18},
            "City": {"strategy": "most_frequent"}
        }
        """
        self.columns = columns
        self.fill_values_: dict[str, object] = {}

    def fit(self, df: pd.DataFrame, **kwargs) -> "NaPreprocessing":
        for col, cfg in self.columns.items():
            if col not in df.columns:
                raise ValueError(f"Column {col} not found in dataframe.")

            strategy = cfg.get("strategy")
            if strategy not in ["mean", "median", "most_frequent", "constant"]:
                raise ValueError(f"Invalid strategy {strategy} for column {col}.")

            if strategy == "mean":
                self.fill_values_[col] = df[col].mean()
            elif strategy == "median":
                self.fill_values_[col] = df[col].median()
            elif strategy == "most_frequent":
                self.fill_values_[col] = df[col].mode(dropna=True)[0]
            elif strategy == "constant":
                if "const_value" not in cfg:
                    raise ValueError(f"Constant strategy requires 'const_value' for {col}.")
                self.fill_values_[col] = cfg["const_value"]

        return self

    def transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df_copy = df.copy()
        for col, fill_val in self.fill_values_.items():
            df_copy[col] = df_copy[col].fillna(fill_val)
        return df_copy

    def fit_transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return super().fit_transform(df, **kwargs)
