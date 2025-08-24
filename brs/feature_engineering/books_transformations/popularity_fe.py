import pandas as pd
import numpy as np

from ..feature_engineering import FeatureEngineeringStrategy

class PopularityFeature(FeatureEngineeringStrategy):
    def __init__(self, ratings_df: pd.DataFrame, log:bool = True):
        self.ratings_df = ratings_df
        self.log = log
    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        popularity = self.ratings_df["ISBN"].value_counts()

        df["Popularity-Edition"] = df["ISBN"].map(
            lambda ISBNS: sum([popularity.get(ISBN, 0) for ISBN in ISBNS])
        ).astype(int)
        if self.log:
            df["Popularity-Edition-Log"] = np.log1p(df["Popularity-Edition"])

        df["Popularity-Book"] = df["Same-Book"].map(
            lambda ISBNS: sum([popularity.get(ISBN, 0) for ISBN in ISBNS])
        ).astype(int)
        if self.log:
            df["Popularity-Book-Log"] = np.log1p(df["Popularity-Book"])

        return df
