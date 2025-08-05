from abc import ABC, abstractmethod
import pandas as pd

class LoadDataset(ABC):
    @abstractmethod
    def load_dataset(self, dataset_name: str) -> pd.DataFrame:
        """
        Loads a dataset from a source and returns it as a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            The loaded dataset
        """
        pass
