import pandas as pd
 
from brs.pipelines.dataframe import DataframePipelineStep

class SetDefaultTypes(DataframePipelineStep):
    def fit(self, df: pd.DataFrame, **kwargs) -> "DataframePipelineStep":
        """Fits the step to the data.

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
        Transforms the given DataFrame by dropping unnecessary columns, setting the
        correct dtypes for the columns, and removing any rows with non-numeric
        values in the "Year-Of-Publication" column.

        Args:
            df: the DataFrame to transform
            **kwargs: any additional keyword arguments

        Returns:
            the transformed DataFrame
        """
        
        df = df.copy()

        df.drop(columns=["Image-URL-S", "Image-URL-M", "Image-URL-L"], inplace=True, errors="ignore")

        df = df[df["Year-Of-Publication"].astype(str).str.isnumeric()]
        df["Year-Of-Publication"] = df["Year-Of-Publication"].astype(int)

        df = df.astype({
            "ISBN": "string",
            "Book-Title": "string",
            "Book-Author": "string",
            "Publisher": "string",
            "Year-Of-Publication": "Int64"
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
