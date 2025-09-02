import pandas as pd

import brs.config as config
from brs.pipelines.dataframe import DataframePipelineStep

class ClearifyPublicationYear(DataframePipelineStep):
    def fit(self, df: pd.DataFrame, **kwargs) -> "DataframePipelineStep":
        """
        Fits the step to the data.

        This step does not fit any parameters from the data, it simply returns
        itself.

        Args:
            df: the DataFrame to fit the step to
            **kwargs: any additional keyword arguments

        Returns:
            self
        """
        return self
    
    def transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:  
        """
        Clearify the "Year-Of-Publication" column by setting all values that are either 0 or greater than the current year to NaN.
        Current year is got from config

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe containing the "Year-Of-Publication" column.
        
        Returns
        -------
        pd.DataFrame
            The dataframe with the "Year-Of-Publication" column cleared.
        """

        df = df.copy()

        df.loc[(df["Year-Of-Publication"] == 0)
                | (df["Year-Of-Publication"] > config.dataset_current_year), "Year-Of-Publication"] = pd.NA
        
        df["Year-Of-Publication"] = df["Year-Of-Publication"].astype("Int64")

        return df 

    def fit_transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Fits the step to the data and transforms it.

        This is a convenience function that is equivalent to calling
        `fit` followed by `transform`. It is useful for when you want
        to fit the step to the data and transform it in a single step.

        Args:
            df: the DataFrame to fit and transform
            **kwargs: any additional keyword arguments

        Returns:
            the transformed DataFrame
        """
        return super().fit_transform(df, **kwargs)
