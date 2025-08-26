import pandas as pd

from brs.preprocessing import UsersClarifyStrategy, BooksClarifyStrategy

class BaselinePipeline:
    def __init__(self):
        self.data_users = None
        self.data_ratings = None
        self.data_books = None
        self.isbn2rating = None

    def fit(
        self, 
        data_users: pd.DataFrame, 
        data_ratings: pd.DataFrame, 
        data_books: pd.DataFrame
    ) -> "BaselinePipeline":
        self.data_users = data_users.copy()
        self.data_ratings = data_ratings.copy()
        self.data_books = data_books.copy()

        self.data_users = UsersClarifyStrategy(ratings_df=self.data_ratings).preprocess(self.data_users)
        self.data_books = BooksClarifyStrategy(
            ratings_df=self.data_ratings, 
            clear_users_df=self.data_users
        ).preprocess(self.data_books)

        books_exploded = self.data_books.explode("ISBN")
        self.isbn2rating = dict(zip(books_exploded["ISBN"], books_exploded["Rating-Book"]))

        return self

    def transform(self, data_ratings: pd.DataFrame) -> pd.DataFrame:
        if self.isbn2rating is None:
            raise ValueError("Pipeline should be fitted first")

        data_ratings = data_ratings.copy()
        data_ratings["Mean-Book-Rating"] = data_ratings["ISBN"].map(self.isbn2rating)
        return data_ratings
