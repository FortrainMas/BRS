import pandas as pd

from brs.pipelines.dataframe import DataframePipelineStep

class RemoveAgeDropouts(DataframePipelineStep):
    def __init__(self, min_age_threshold: int = 10, max_age_threshold: int = 100):
        """
        Initializes the RemoveAgeDropouts step.

        Args:
            min_age_threshold: the minimum age that a user must be to be considered
                as a valid user.
            max_age_threshold: the maximum age that a user can be to be considered
                as a valid user.
        """
        self.min_age_threshold = min_age_threshold
        self.max_age_threshold = max_age_threshold

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
        Removes users with an age below min_age_threshold or above max_age_threshold.
        
        Args:
            df: the DataFrame to transform
            **kwargs: any additional keyword arguments
                min_age_threshold: overrides the minimum age threshold
                max_age_threshold: overrides the maximum age threshold
        
        Returns:
            the transformed DataFrame
        """

        df = df.copy()

        min_ratings_threshold = kwargs.get("min_age_threshold", self.min_age_threshold)
        max_ratings_threshold = kwargs.get("max_age_threshold", self.max_age_threshold)

        df["Age"] = df["Age"].where(
            (df["Age"] >= min_ratings_threshold) & (df["Age"] <= max_ratings_threshold),
            pd.NA
        )

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
