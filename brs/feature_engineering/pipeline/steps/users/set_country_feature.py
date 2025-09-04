import pandas as pd

from brs.pipelines.dataframe import DataframePipelineStep

class UserSetCountryFeature(DataframePipelineStep):
    def fit(self, df: pd.DataFrame, **kwargs) -> "DataframePipelineStep":
        return self
    
    def transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = df.copy()

        if "Location" not in df.columns:
            raise ValueError("Column 'Location' not found in the dataframe.")

        df["Country"] = df["Location"].apply(
            lambda x: x.split(",")[-1] or None
        )

        return df

    def fit_transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return super().fit_transform(df, **kwargs)
