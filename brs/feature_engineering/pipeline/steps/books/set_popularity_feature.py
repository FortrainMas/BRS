import pandas as pd
import numpy as np

from brs.pipelines.dataframe import DataframePipelineStep

class BookSetPopularityFeature(DataframePipelineStep):
    def __init__(self, ratings_df: pd.DataFrame, log:bool=True):
        self.ratings_df = ratings_df
        self.log = log
        self.mappings = None

    def fit(self, df: pd.DataFrame, **kwargs) -> "DataframePipelineStep":
        self.ratings_df = kwargs.get("ratings_df", self.ratings_df)

        grouped = self.ratings_df.groupby("ISBN")["Book-Rating"].count()
        self.mappings = grouped.to_dict()

        return self
    
    def transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        if self.mappings is None:
            raise ValueError("Please fit the pipeline first.")

        log  = kwargs.get("log", self.log)

        df = df.copy()
        df["Popularity"] = df["ISBN"].map(self.mappings)

        if log:
            df["Popularity_log"] = np.log1p(df["Popularity"])

        return df

    def fit_transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        self.fit(df, **kwargs)
        return self.transform(df, **kwargs)
