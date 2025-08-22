import pandas as pd

from ..feature_engineering import FeatureEngineeringStrategy

class AgeGroupsFeature(FeatureEngineeringStrategy):
    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        if "Age" not in df.columns:
            raise ValueError("Column 'Age' not found in the dataframe.")

        bins = [0, 12, 18, 30, 45, 60, 100]
        labels = ["0-12", "13-17", "18-29", "30-44", "45-59", "60+"]

        df["Age_group"] = pd.cut(df["Age"], bins=bins, labels=labels, right=True, include_lowest=True)
        return df
