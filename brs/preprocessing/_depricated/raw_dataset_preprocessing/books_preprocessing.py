import pandas as pd

from ..preprocessing import PreprocessingStrategy
from ..books_preprocessing import BooksPreprocessingStrategy
from .users_preprocessing import UsersClarifyStrategy
from brs.feature_engineering import PopularityFeature, RatingFeature, ElderityFeature


class BooksClarifyStrategy(PreprocessingStrategy):
    def __init__(self, ratings_df: pd.DataFrame, clear_users_df: pd.DataFrame, train: bool = True):
        self.ratings_df = ratings_df
        self.clear_users_df = clear_users_df
        self.train = train

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        data_books = df.copy()
        data_ratings = self.ratings_df
        data_users = self.clear_users_df

        if "Mean-Rating" not in data_users.columns:
            raise ValueError("Users data frame should be processed before")

        data_users = UsersClarifyStrategy(ratings_df = data_ratings).preprocess(data_users)
        data_ratings = data_ratings[data_ratings["User-ID"].isin(data_users["User-ID"])]

        
        data_books = BooksPreprocessingStrategy().preprocess(data_books)
        data_books = ElderityFeature(ratings_df=data_ratings).apply_transformation(data_books)
        if self.train:
            data_books = PopularityFeature(ratings_df=data_ratings).apply_transformation(data_books)
            data_books = RatingFeature(ratings_df=data_ratings).apply_transformation(data_books)

        return data_books
