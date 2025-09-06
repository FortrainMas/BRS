import pandas as pd

from brs.pipelines.dataframe import DataframePipelineStep

class RemoveZerolikers(DataframePipelineStep):
    def __init__(self, zero_likers_threshold:int = 5):
        """
        Initializes the RemoveZerolikers step.

        Args:
            zero_likers_threshold: the minimum number of ratings a user must have
                to be considered as a valid user and not a "zeroliker".
        """

        self.zero_likers_threshold = zero_likers_threshold

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
        Removes users with no non-zero ratings and less than the threshold number of ratings.

        Args:
            df: the DataFrame to transform
            **kwargs: any additional keyword arguments
                zero_likers_threshold: overrides the minimum number of ratings threshold

        Returns:
            the transformed DataFrame
        """

        df = df.copy()

        zero_likers_threshold = kwargs.get("zero_likers_threshold", self.zero_likers_threshold)

        return df.groupby("User-ID").filter(
            lambda x: x["Book-Rating"].sum() != 0 and x["Book-Rating"].count() > zero_likers_threshold
        )
    
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
    