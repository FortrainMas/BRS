import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi

import brs.config as config
from .load_dataset import LoadDataset

class LoadLocalDataset(LoadDataset):
    def load_dataset(self, dataset_name: str) -> pd.DataFrame:
        """
        Loads a dataset from the local file system and returns it as a pandas DataFrame.
        In case if it is not downloaded from Kaggle downloads it

        Returns
        -------
        pd.DataFrame
            The loaded dataset
        """
        if dataset_name != "books" and dataset_name != "ratings" and dataset_name != "users":
            raise ValueError("Invalid dataset name: must be 'books', 'ratings', or 'users'") 
        
        dataset_path = os.path.join(config.root_dir, "data", "raw", "kaggle-books", f"{dataset_name}.csv")
        
        if not Path(dataset_path).exists():
            dataset_dir = os.path.join(config.root_dir, "data", "raw", "kaggle-books")
            if Path(dataset_dir).exists():
                raise FileNotFoundError(f"Dataset '{dataset_name}' not found but {dataset_dir} exists. To prevent accidental deletion it won't be downloaded from Kaggle")
            else:
                load_dotenv()
                api = KaggleApi()
                api.authenticate()

                dataset_name = "arashnic/book-recommendation-dataset"
                local_dir = dataset_dir
                Path(local_dir).mkdir(parents=True, exist_ok=True)
                api.dataset_download_files(dataset_name, path=local_dir, unzip=True)

        Path(dataset_path).chmod(0o777)            
        return pd.read_csv(dataset_path)
        