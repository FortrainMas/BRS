import pandas as pd
from brs.pipelines.dataframe import DataframePipelineStep

class BookSetRatingFeature(DataframePipelineStep):
    def __init__(self, ratings_df: pd.DataFrame):
        self.ratings_df = ratings_df
        self.mappings = None

    def fit(self, df: pd.DataFrame, **kwargs) -> "DataframePipelineStep":
        self.ratings_df = kwargs.get("ratings_df", self.ratings_df)

        grouped = self.ratings_df.groupby("ISBN")["Book-Rating"].mean()
        self.mappings = grouped.to_dict()

        return self
    
    def transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        if self.mappings is None:
            raise ValueError("Please fit the pipeline first.")

        df = df.copy()
        df["Mean-Rating"] = df["ISBN"].map(self.mappings)
        return df

    def fit_transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        self.fit(df, **kwargs)
        return self.transform(df, **kwargs)
