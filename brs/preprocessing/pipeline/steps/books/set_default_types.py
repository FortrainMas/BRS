import pandas as pd
 
from brs.pipelines.dataframe import DataframePipelineStep

class SetDefaultTypes(DataframePipelineStep):
    def fit(self, df: pd.DataFrame, **kwargs) -> "DataframePipelineStep":
        return self

    def transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
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
        return super().fit_transform(df, **kwargs)