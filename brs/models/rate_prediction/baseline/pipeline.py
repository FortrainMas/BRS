import pandas as pd

from brs.pipelines.dataframe import DataframePipeline
from brs.preprocessing.pipeline import RatingsRemoveZeros

class BaselineRatePipeline(DataframePipeline):
    def __init__(self):
        steps = [
            RatingsRemoveZeros()
        ]
        super().__init__(steps)
