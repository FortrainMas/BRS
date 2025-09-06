import pandas as pd

from brs.pipelines.dataframe import DataframePipeline
from brs.preprocessing.pipeline import RatingsRemoveZeros
from brs.preprocessing.pipeline import UsersRemoveUnrated

class RatingsModelPipeline(DataframePipeline):
    def __init__(self):
        steps = [
            RatingsRemoveZeros()
        ]
        super().__init__(steps)


class UsersModelPipeline(DataframePipeline):
    def __init__(self, ratings_df: pd.DataFrame):
        steps = [
            UsersRemoveUnrated(ratings_df=ratings_df)
        ]
        super().__init__(steps)
