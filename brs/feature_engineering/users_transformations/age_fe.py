import pandas as pd

from ..feature_engineering import FeatureEngineeringStrategy

class AgeGroupsFeature(FeatureEngineeringStrategy):
    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates a new column "Age_group" that contains the age group from the "Age" column.

        The age groups are as follows:
        - 0-12
        - 13-17
        - 18-29
        - 30-44
        - 45-59
        - 60+

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe containing the "Age" column.

        Returns
        -------
        pd.DataFrame
            The dataframe with the new "Age_group" column.
        """
        if "Age" not in df.columns:
            raise ValueError("Column 'Age' not found in the dataframe.")

        bins = [0, 12, 17, 29, 44, 59, 200]
        labels = ["0-12", "13-17", "18-29", "30-44", "45-59", "60+"]

        df["Age_group"] = pd.cut(df["Age"], bins=bins, labels=labels, include_lowest=True)
        return df
