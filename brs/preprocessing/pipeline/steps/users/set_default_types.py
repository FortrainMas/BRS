import pandas as pd
 
from brs.pipelines.dataframe import DataframePipelineStep

class SetDefaultTypes(DataframePipelineStep):
    def fit(self, df: pd.DataFrame, **kwargs) -> "DataframePipelineStep":
        return self

    def transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = df.copy()

        df = df.astype({
            "User-ID": "int64",
            "Location": "string",
            "Age": "Float64"
            })

        return df

    def fit_transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return super().fit_transform(df, **kwargs)
