import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi

import brs.config as config
from .load_dataset import LoadDataset

class LoadLocalDataset(LoadDataset):
    def load_dataset(self, dataset_name: str, clear_version: bool = False) -> pd.DataFrame:
        """
        Loads a dataset from a local file and returns it as a pandas DataFrame.

        If the requested dataset does not exist in the raw data directory, it will be downloaded from Kaggle
        if the user has a Kaggle account and the raw data directory is empty.

        If the clear version is requested, the dataset from the clear data directory will be used if it exists.

        Returns
        -------
        pd.DataFrame
            The loaded dataset
        """
        
        if dataset_name != "books" and dataset_name != "ratings" and dataset_name != "users":
            raise ValueError("Invalid dataset name: must be 'books', 'ratings', or 'users'") 
        
        if clear_version:
            dataset_path = os.path.join(config.root_dir, "data", "clear", "kaggle-books", f"{dataset_name}.csv")
            if Path(dataset_path).exists():
                Path(dataset_path).chmod(0o777)
                return pd.read_csv(dataset_path)
            else:
                raise ValueError(f"Clear version for dataset '{dataset_name}' not found in '{dataset_path}'")
        
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
        