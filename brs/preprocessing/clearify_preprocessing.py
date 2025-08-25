import pandas as pd

from .preprocessing import PreprocessingStrategy
from brs.preprocessing import UsersPreprocessingStrategy, BooksPreprocessingStrategy
from brs.feature_engineering import AgeGroupsFeature, CountryFeature, MeanRatingFeature, NumRatingsFeature
from brs.feature_engineering import PopularityFeature, RatingFeature, ElderityFeature


class ClearifyPreporcessingStrategy(PreprocessingStrategy):
    def __init__(self, user_df:pd.DataFrame, 
                 books_df:pd.DataFrame, 
                 minimum_age:int = 5, 
                 maximum_age:int = 90, 
                 zero_likers_threshold:int = 5, 
                 maximum_rates_threshold:int=100):
        self.user_df = user_df
        self.books_df = books_df
        self.minimum_age = minimum_age
        self.maximum_age = maximum_age
        self.zero_likers_threshold = zero_likers_threshold
        self.maximum_rates_threshold = maximum_rates_threshold
        

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        data_ratings = df
        data_users = self.user_df
        data_books = self.books_df

        data_users = UsersPreprocessingStrategy().preprocess(data_users)
        data_users = AgeGroupsFeature().apply_transformation(data_users)
        data_users = CountryFeature().apply_transformation(data_users)
        data_users = MeanRatingFeature(ratings_df=data_ratings).apply_transformation(data_users)
        data_users = NumRatingsFeature(ratings_df=data_ratings).apply_transformation(data_users)


        threshold = self.zero_likers_threshold
        mask = (data_users["Mean-Rating"] == 0) & (data_users["Number-Ratings"] > threshold)
        data_users = data_users[~mask]

        threshold = self.maximum_rates_threshold
        mask = (data_users["Number-Ratings"] == 0) | (data_users["Number-Ratings"] > threshold)
        data_users = data_users[~mask]


        data_books = BooksPreprocessingStrategy().preprocess(data_books)
        data_books = PopularityFeature(ratings_df=data_ratings).apply_transformation(data_books)
        data_books = RatingFeature(ratings_df=data_ratings).apply_transformation(data_books)
        data_books = ElderityFeature(ratings_df=data_ratings).apply_transformation(data_books)

        return data_ratings