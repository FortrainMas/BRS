import pandas as pd

from ..feature_engineering import FeatureEngineeringStrategy

class CountryFeature(FeatureEngineeringStrategy):
    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates a new column "Countries" that contains the country from the "Location" column.

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe containing the "Location" column.

        Returns
        -------
        pd.DataFrame
            The dataframe with the new "Countries" column.
        """
        if "Location" not in df.columns:
            raise ValueError("Column 'Location' not found in the dataframe.")

        df["Country"] = df["Location"].apply(
            lambda x: x.split(" ")[-1].rstrip(",") or None
        )
        return df
