import pandas as pd

from brs.preprocessing import UsersClarifyStrategy, BooksClarifyStrategy

class LinRegPipeline:
    def __init__(self):
        pass

    def top_dummies(self, df: pd.DataFrame, col: str, top_n: int = 100, prefix: str = None) -> pd.DataFrame:
        prefix = prefix or col
        top = df[col].value_counts().nlargest(top_n).index
        df[col] = df[col].where(df[col].isin(top), other="Other")
        return pd.get_dummies(df, columns=[col], prefix=prefix, drop_first=True)


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


        data_users.drop(["Location", "Age"], axis=1, inplace=True)
        data_users["Age-Group"] = data_users["Age-Group"].cat.add_categories(["Unknown"])
        data_users["Age-Group"] = data_users["Age-Group"].fillna("Unknown")
        data_users = pd.get_dummies(data_users, columns=["Age-Group"], prefix="Age-Group", drop_first=True)
        data_users["Country"] = data_users["Country"].fillna("Unknown")
        data_users = pd.get_dummies(data_users, columns=["Country"], prefix="Country", drop_first=True)

        data_books.drop(["Year-Of-Publication", "Book-Title", "Same-Book"], axis=1, inplace=True)
        data_books["Book-Author"] = data_books["Book-Author"].fillna("Unknown")
        data_books["Publisher"] = data_books["Publisher"].fillna("Unknown")
        data_books = self.top_dummies(data_books, "Book-Author", top_n=100, prefix="Book-Author")
        data_books = self.top_dummies(data_books, "Publisher", top_n=100, prefix="Publisher")


        cols = [
            "Popularity-Edition",
            "Popularity-Edition-Log",
            "Popularity-Book",
            "Popularity-Book-Log",
            "Rating-Edition",
            "Rating-Book",
        ]
        for col in cols:
            data_books[col] = data_books[col].fillna(-1)

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
