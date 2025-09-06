from typing import Sequence

import pandas as pd

from brs.pipelines.dataframe.dataframe_pipeline_step import DataframePipelineStep

class DataframePipeline(DataframePipelineStep):
    def __init__(self, steps: Sequence[DataframePipelineStep]):
        self.steps = steps

    def fit(self, df: pd.DataFrame, **kwargs) -> "DataframePipeline":
        local_df = df.copy()
        for i in range(len(self.steps)):
            self.steps[i].fit(local_df, **kwargs)
            local_df = self.steps[i].transform(local_df, **kwargs)
        return self

    def transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        local_df = df.copy()
        for i in range(len(self.steps)):
            local_df = self.steps[i].transform(local_df, **kwargs)
        return local_df
    
    def fit_transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        local_df = df.copy()
        for i in range(len(self.steps)):
            self.steps[i].fit(local_df, **kwargs)
            local_df = self.steps[i].transform(local_df, **kwargs)
        return local_df
