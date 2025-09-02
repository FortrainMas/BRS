from .preprocessing import PreprocessingStrategy

import pandas as pd

class UsersPreprocessingStrategy(PreprocessingStrategy):
    def __init__(self, minimum_age:int = 5, maximum_age:int = 90):
        self.minimum_age = minimum_age
        self.maximum_age = maximum_age

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        if "Age" not in df.columns or "Location" not in df.columns:
            raise ValueError("Column 'Age' or 'Location' not found in the dataframe.")


        df = df.astype({"Age": "Int64", "Location": "string"})

        df.loc[(df["Age"] < self.minimum_age) | (df["Age"] > self.maximum_age),"Age"] = None

        return df
