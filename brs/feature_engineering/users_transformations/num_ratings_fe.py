import pandas as pd

from ..feature_engineering import FeatureEngineeringStrategy

class NumRatingsFeature(FeatureEngineeringStrategy):
    def __init__(self, ratings_df:pd.DataFrame):
        self.ratings_df = ratings_df

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        ratings_df = self.ratings_df.groupby("User-ID")["Book-Rating"].count()


        df["Number-Ratings"] = df["User-ID"].map(ratings_df).fillna(0).astype("Int64")

        return df
