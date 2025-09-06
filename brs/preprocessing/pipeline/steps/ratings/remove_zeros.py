from pandas import DataFrame
from brs.pipelines.dataframe import DataframePipelineStep

class RemoveZeros(DataframePipelineStep):
    def fit(self, df: DataFrame, **kwargs) -> "DataframePipelineStep":
        return self

    def transform(self, df: DataFrame, **kwargs) -> DataFrame:
        return df[df["Book-Rating"] != 0]
    
    def fit_transform(self, df: DataFrame, **kwargs) -> DataFrame:
        return super().fit_transform(df, **kwargs)
