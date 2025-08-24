import pandas as pd

from ..feature_engineering import FeatureEngineeringStrategy

class MeanRatingFeature(FeatureEngineeringStrategy):
    def __init__(self, ratings_df:pd.DataFrame):
        self.ratings_df = ratings_df

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        ratings_df = self.ratings_df.groupby("User-ID")["Book-Rating"].mean()


        df["Mean-Rating"] = df["User-ID"].map(ratings_df).astype("float64")

        return df
