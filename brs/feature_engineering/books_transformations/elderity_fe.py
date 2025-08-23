import logging

import pandas as pd
import numpy as np

from ..feature_engineering import FeatureEngineeringStrategy
import brs.config as config

class ElderityFeature(FeatureEngineeringStrategy):
    def __init__(self, ratings_df: pd.DataFrame, log:bool = True):
        self.ratings_df = ratings_df
        self.log = log
    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df["Elderity"] = config.dataset_current_year - df["Year-Of-Publication"]
        if self.log:
            df["Elderity_log"] = np.log1p(df["Elderity"])
        return df
