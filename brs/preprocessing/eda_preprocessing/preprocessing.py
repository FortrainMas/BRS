from abc import ABC, abstractmethod

import pandas as pd

class PreprocessingStrategy(ABC):
    @abstractmethod
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform a specific type of preprocessing on the given dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe to be preprocessed.

        Returns:
        pd.DataFrame: The preprocessed dataframe.
        """

        pass
