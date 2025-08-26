import pandas as pd

from brs.preprocessing import UsersClarifyStrategy, BooksClarifyStrategy

class LinRegPipeline:
    def __init__(self):
        pass

    def fit(
        self, 
        data_users: pd.DataFrame, 
        data_ratings: pd.DataFrame, 
        data_books: pd.DataFrame
    ) -> "LinRegPipeline":
        self.data_users = data_users.copy()
        self.data_ratings = data_ratings.copy()
        self.data_books = data_books.copy()

        self.data_users = UsersClarifyStrategy(ratings_df=self.data_ratings).preprocess(self.data_users)
        self.data_books = BooksClarifyStrategy(
            ratings_df=self.data_ratings, 
            clear_users_df=self.data_users
        ).preprocess(self.data_books)


        self.data_users.drop(["Location", "Age"], axis=1, inplace=True)
        self.data_users["Age-Group"].fillna("Unknown", inplace=True)
        self.data_users = pd.get_dummies(self.data_users, columns=["Age-Group"], prefix="Age-Group", drop_first=True)
        self.data_users["Country"].fillna("Unknown", inplace=True)
        self.data_users = pd.get_dummies(self.data_users, columns=["Country"], prefix="Country", drop_first=True)


        self.data_books.drop(["Year-Of-Publication", "Book-Title", "Same-Book"], axis=1, inplace=True)
        self.data_books["Book-Author"].fillna("Unknown", inplace=True)
        self.data_books = pd.get_dummies(self.data_books, columns=["Book-Author"], prefix="Book-Author", drop_first=True)
        self.data_books["Publisher"].fillna("Unknown", inplace=True)
        self.data_books = pd.get_dummies(self.data_books, columns=["Publisher"], prefix="Publisher", drop_first=True)
        cols = [
            "Popularity-Edition",
            "Popularity-Edition-Log",
            "Popularity-Book",
            "Popularity-Book-Log",
            "Rating-Edition",
            "Rating-Book",
        ]

        
        self.data_books_exploded = self.data_books.explode("ISBN")

        self.data_books[cols] = self.data_books[cols].fillna(-1)

        return self

    def transform(self, data_ratings: pd.DataFrame) -> pd.DataFrame:
        if self.data_users is None:
            raise ValueError("Pipeline should be fitted first")

        data_ratings = data_ratings.copy()

        merged = data_ratings.merge(
            self.data_users,
            on="User-ID",
            how="left"
        )

        merged = merged.merge(
            self.data_books_exploded,
            on="ISBN",
            how="left"
        )

        merged = merged.fillna(merged.mean(numeric_only=True))

        return data_ratings
