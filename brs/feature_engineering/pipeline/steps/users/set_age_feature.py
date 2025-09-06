import pandas as pd

from brs.pipelines.dataframe import DataframePipelineStep

class UserSetAgeFeature(DataframePipelineStep):
    def fit(self, df: pd.DataFrame, **kwargs) -> "DataframePipelineStep":
        return self
    
    def transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = df.copy()

        if "Age" not in df.columns:
            raise ValueError("Column 'Age' not found in the dataframe.")

        bins = [0, 12, 17, 29, 44, 59, 200]
        labels = ["0-12", "13-17", "18-29", "30-44", "45-59", "60+"]

        df["Age-Group"] = pd.cut(df["Age"], bins=bins, labels=labels, include_lowest=True)
        return df

    def fit_transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return super().fit_transform(df, **kwargs)
    