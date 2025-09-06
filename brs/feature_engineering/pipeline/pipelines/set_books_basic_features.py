import pandas as pd

from brs.pipelines.dataframe import DataframePipeline
from brs.feature_engineering.pipeline import BookSetRatingFeature, BookSetElderityFeature, BookSetPopularityFeature


class SetBooksBasicFeatures(DataframePipeline):
    def __init__(self, ratings_df: pd.DataFrame, log:bool=True):
        steps = [
            BookSetRatingFeature(ratings_df=ratings_df), 
            BookSetElderityFeature(log=log), 
            BookSetPopularityFeature(ratings_df=ratings_df, log=log)
        ]
        
        super().__init__(steps)
        self.ratings_df = ratings_df
        self.log = log
