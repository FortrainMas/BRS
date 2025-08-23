from .preprocessing import PreprocessingStrategy

import pandas as pd

import brs.config as config

class BooksPreprocessingStrategy(PreprocessingStrategy):
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Drop unnecessary columns
        df.drop(columns=["Image-URL-S", "Image-URL-M", "Image-URL-L"], inplace=True)

        # Clean up the "Year-Of-Publication" column
        df = df[df["Year-Of-Publication"].astype(str).str.isnumeric()]
        df["Year-Of-Publication"] = df["Year-Of-Publication"].astype(int)
        df.loc[(df["Year-Of-Publication"] == 0) | (df["Year-Of-Publication"] > config.dataset_current_year)] = None

        # Modify ISBN column
        df["ISBN"] = df["ISBN"].apply(lambda x: [x])

        # Set types
        df = df.astype({
            "Book-Title": "string",
            "Book-Author": "string",
            "Publisher": "string",
            "Year-Of-Publication": "Int64"
            })


        # Remove duplicates
        df = (
            df.groupby(["Book-Title", "Book-Author", "Publisher", "Year-Of-Publication"], as_index=False)
              .agg({
                  "ISBN": lambda x: sorted(set(sum(x, [])))
              })
        )

        # Set reference to the same book
        same_book_map = (
            df.groupby(["Book-Title", "Book-Author"])["ISBN"]
              .apply(lambda x: sorted(set(sum(x, []))))
              .reset_index(name="Same-Book")
        )
        
        df = df.merge(same_book_map, on=["Book-Title", "Book-Author"], how="left")
        return df
