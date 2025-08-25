import pandas as pd

from ..preprocessing import PreprocessingStrategy
from ..users_preprocessing import UsersPreprocessingStrategy
from brs.feature_engineering import AgeGroupsFeature, CountryFeature, MeanRatingFeature, NumRatingsFeature


class UsersClarifyStrategy(PreprocessingStrategy):
    def __init__(self, 
                 ratings_df:pd.DataFrame,
                 minimum_age:int = 5, 
                 maximum_age:int = 90, 
                 zero_likers_threshold:int = 5, 
                 maximum_rates_threshold:int=100):
        self.ratings_df = ratings_df
        self.minimum_age = minimum_age
        self.maximum_age = maximum_age
        self.zero_likers_threshold = zero_likers_threshold
        self.maximum_rates_threshold = maximum_rates_threshold
        

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        data_ratings = self.ratings_df
        data_users = df

        data_users = UsersPreprocessingStrategy().preprocess(data_users)
        data_users = AgeGroupsFeature().apply_transformation(data_users)
        data_users = CountryFeature().apply_transformation(data_users)
        data_users = MeanRatingFeature(ratings_df=data_ratings).apply_transformation(data_users)
        data_users = NumRatingsFeature(ratings_df=data_ratings).apply_transformation(data_users)

        mask = (data_users["Mean-Rating"] == 0) & (data_users["Number-Ratings"] > self.zero_likers_threshold)
        data_users = data_users[~mask]

        mask = (data_users["Number-Ratings"] == 0) | (data_users["Number-Ratings"] > self.maximum_rates_threshold)
        data_users = data_users[~mask]


        return data_users
