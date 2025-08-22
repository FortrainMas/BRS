import os
import shutil
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi

import brs.config as config
from .load_dataset import LoadDataset

class LoadKaggleDataset(LoadDataset):
    def load_dataset(self, dataset_name: str) -> pd.DataFrame:
        """
        Loads a dataset from Kaggle and returns it as a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            The loaded dataset
        """
        if dataset_name != "books" and dataset_name != "ratings" and dataset_name != "users":
            raise ValueError("Invalid dataset name: must be 'books', 'ratings', or 'users'") 

        load_dotenv()
        api = KaggleApi()
        api.authenticate()

        kaggle_dataset_name = "arashnic/book-recommendation-dataset"
        local_dir = os.path.join(config.root_dir, "tmp", "kaggle-books")
        Path(local_dir).mkdir(parents=True, exist_ok=True)

        api.dataset_download_files(kaggle_dataset_name, path=local_dir, unzip=True)

        df = pd.read_csv(os.path.join(local_dir, f"{dataset_name}.csv"))
        shutil.rmtree(os.path.join(config.root_dir, "tmp"))
        return df
