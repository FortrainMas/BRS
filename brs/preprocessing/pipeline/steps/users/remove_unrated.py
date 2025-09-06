import pandas as pd

from brs.pipelines.dataframe import DataframePipelineStep

class RemoveUnrated(DataframePipelineStep):
    def __init__(self, ratings_df: pd.DataFrame):
        """
        Initializes the RemoveUnrated step.

        Args:
            ratings_df: the DataFrame of ratings to use for filtering
        """
        self.ratings_df = ratings_df

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
        Removes users from the dataframe who have not rated any books.

        Args:
            df: the DataFrame to transform
            **kwargs: any additional keyword arguments
                ratings_df: the DataFrame of ratings to use for filtering

        Returns:
            the transformed DataFrame
        """

        df = df.copy()
        ratings_df = kwargs.get("ratings_df", self.ratings_df)
        return df[df["User-ID"].isin(ratings_df["User-ID"])]
    
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
