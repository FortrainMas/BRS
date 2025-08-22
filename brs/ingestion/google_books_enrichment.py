import os
import json
import logging

import pandas as pd
import requests
from pathlib import Path

from .load_dataset import LoadDataset
from .load_local import LoadLocalDataset
import brs.config as config
from brs.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

class GoogleBooksEnrichment(LoadDataset):
    def __init__(self,
                 download=True, 
                 save_rate:int = 100, 
                 max_results:int = 10,
                 limit:int|None = None,
                 reattempts:int = 10):
        
        self.save_rate = save_rate
        self.max_results = max_results
        self.limit = limit
        self.download = download
        self.reattemps = reattempts

        self.total_json : list[dict] = []
        self.json_artifacts_path: str = os.path.join(config.root_dir, "artifacts", "google_books.json")
        self.df_path = os.path.join(config.root_dir, "data", "raw", "kaggle-books", "books_enriched.csv")

    def save(self, df: pd.DataFrame):
        with open(self.json_artifacts_path, "w") as f:
                    f.write(json.dumps(self.total_json))
        df.to_csv(self.df_path, index=False)
        logger.info("Dataframe iteration saved")

    def load_saved_data(self) -> pd.DataFrame:
        try:
            pdf = pd.read_csv(self.df_path)
            return pdf
        except FileNotFoundError:
            logger.warning("No saved dataframe found. It is expected if enrichment is called for the first time")
            df = LoadLocalDataset().load_dataset("books")
            df["enriched"] = False
            return df
        
    def load_dataset(self, dataset_name: str) -> pd.DataFrame:
        if dataset_name != "books":
            raise ValueError("Invalid dataset name: must be 'books'")

        df = self.load_saved_data()
        false_indices = df.index[~df["enriched"]]
        
        if not self.download or len(false_indices) == 0:
            return df

        reattempts = self.reattemps

        start_idx = false_indices[0]
        for i, (idx, row) in enumerate(df.loc[start_idx:start_idx + self.limit].iterrows()):
            if i % self.save_rate == 0:
                self.save(df)
                
            author = row["Book-Author"]
            title = row["Book-Title"]
            
            if not author:
                url = f'https://www.googleapis.com/books/v1/volumes?q=intitle:"{title}"&maxResults={self.max_results}'
            else:
                url = f'https://www.googleapis.com/books/v1/volumes?q=inauthor:"{author}"+intitle:"{title}"&maxResults={self.max_results}'

            try:
                resp = requests.get(url)
                if resp.status_code == 200:
                    reattempts = self.reattemps
                    data = resp.json()
                    items = data.get("items", [])
                    self.total_json.append(items)

                    description = None
                    genre = None
                    
                    for item in items:
                        volume_info = item.get("volumeInfo")
                        volume_description = volume_info.get("description")
                        if volume_description:
                            if not description:
                                description = volume_description
                            elif volume_info.get("title") == title and len(volume_description) > len(description):
                                description = volume_description
                        if not genre and volume_info.get("categories"):
                            genre = volume_info.get("categories")

                    df.at[idx, "description"] = description
                    df.at[idx, "category"] = genre
                    df.at[idx, "enriched"] = True
                else:
                    logger.warning(f"Response status code is not 200. {resp.status_code}")
                    logger.warning(f"{resp.text}")
                    raise Exception()
                
            except Exception as e:
                df.at[idx, "enriched"] = True
                logger.warning(f"Query failed for {title} by {author}; {e}")
                reattempts -= 1
                if reattempts == 0:
                    logger.error(f"Books dataset enrichment stopped due to lack of reattempts")
                    break

        logger.info("Dataset enrichment successfully finished")
        self.save(df)
        return df
