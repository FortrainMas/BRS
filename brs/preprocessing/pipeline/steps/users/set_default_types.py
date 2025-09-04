import pandas as pd
 
from brs.pipelines.dataframe import DataframePipelineStep

class SetDefaultTypes(DataframePipelineStep):
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
        Sets the default dtypes for the columns in the given DataFrame.

        Args:
            df: the DataFrame to transform
            **kwargs: any additional keyword arguments

        Returns:
            the transformed DataFrame
        """

        df = df.copy()

        df = df.astype({
            "User-ID": "int64",
            "Location": "string",
            "Age": "Float64"
            })

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
