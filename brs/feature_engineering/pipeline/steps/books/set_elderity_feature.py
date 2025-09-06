import pandas as pd
import numpy as np

import brs.config as config
from brs.pipelines.dataframe import DataframePipelineStep

class BookSetElderityFeature(DataframePipelineStep):
    def __init__(self, log:bool = True):
        self.log = log

    def fit(self, df: pd.DataFrame, **kwargs) -> "DataframePipelineStep":
        return self
    
    def transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = df.copy()

        log = kwargs.get("log", self.log)

        df["Elderity"] = config.dataset_current_year - df["Year-Of-Publication"]
        if log:
            df["Elderity_log"] = np.log1p(df["Elderity"])
        return df

    def fit_transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return super().fit_transform(df, **kwargs)
    