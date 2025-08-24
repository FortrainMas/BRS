import pandas as pd

from ..feature_engineering import FeatureEngineeringStrategy

class RatingFeature(FeatureEngineeringStrategy):
    def __init__(self, ratings_df: pd.DataFrame):
        self.ratings_df = ratings_df
    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        ratings_df = self.ratings_df.groupby("ISBN")["Book-Rating"].mean()

        def get_rating(ISBNS: list[str]) -> float|None:
            if all([ratings_df.get(ISBN) is None for ISBN in ISBNS]) :
                return None
            return sum([ratings_df.get(ISBN, 0) for ISBN in ISBNS])/len(ISBNS)

        df["Rating-Edition"] = df["ISBN"].map(get_rating).astype("float64")

        df["Rating-Book"] = df["Same-Book"].map(get_rating).astype("float64")

        return df
