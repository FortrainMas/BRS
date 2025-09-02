import pandas as pd

from brs.pipelines.dataframe import DataframePipelineStep

class RemoveDropouts(DataframePipelineStep):
    def __init__(self, min_ratings_threshold: int = 10, max_ratings_threshold: int = 100):
        self.min_ratings_threshold = min_ratings_threshold
        self.max_ratings_threshold = max_ratings_threshold

    def fit(self, df: pd.DataFrame, **kwargs) -> "DataframePipelineStep":
        return self

    def transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = df.copy()

        min_ratings_threshold = kwargs.get("min_ratings_threshold", self.min_ratings_threshold)
        max_ratings_threshold = kwargs.get("max_ratings_threshold", self.max_ratings_threshold)

        return df.groupby("User-ID").filter(
            lambda x: x["Book-Rating"].count() <= max_ratings_threshold and x["Book-Rating"].count() >= min_ratings_threshold
        )

    def fit_transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return super().fit_transform(df, **kwargs)