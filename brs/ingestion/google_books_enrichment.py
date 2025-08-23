import os
import logging
import asyncio

import aiohttp
from tqdm.asyncio import tqdm_asyncio
import pandas as pd
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
                 batch_size:int = 5):
        
        self.save_rate = save_rate
        self.max_results = max_results
        self.limit = limit
        self.download = download
        self.batch_size = batch_size

        self.df_path = os.path.join(config.root_dir, "data", "raw", "kaggle-books", "books_enriched.csv")

    def save(self, df: pd.DataFrame):
        df.to_csv(self.df_path, index=False)

    def load_saved_data(self) -> pd.DataFrame:
        try:
            pdf = pd.read_csv(self.df_path)
            return pdf
        except FileNotFoundError:
            logger.warning("No saved dataframe found. It is expected if enrichment is called for the first time")
            df = LoadLocalDataset().load_dataset("books")
            df["enriched"] = False
            return df
        
    async def fetch_book(self, session: aiohttp.ClientSession, author:str, title:str) -> list[dict]:
        if not author:
            url = f'https://www.googleapis.com/books/v1/volumes?q=intitle:"{title}"&maxResults={self.max_results}'
        else:
            url = f'https://www.googleapis.com/books/v1/volumes?q=inauthor:"{author}"+intitle:"{title}"&maxResults={self.max_results}'

        reattempt = 30
        while reattempt:
            async with session.get(url) as resp:
                try:
                    if resp.status == 200:
                        data = await resp.json()
                        return data.get("items", [])
                    elif resp.status==409:
                        await asyncio.sleep(1)
                        reattempt-=1
                    else:
                        return []
                except Exception:
                    return []
            
    async def enrich_batch(self, session: aiohttp.ClientSession, batch_df:pd.DataFrame) -> bool:
        tasks = [
            self.fetch_book(session, row["Book-Author"], row["Book-Title"])
            for _, row in batch_df.iterrows()
        ]
        results = await asyncio.gather(*tasks)

        for idx, items in zip(batch_df.index, results):
            description = None
            genre = None
            for item in items:
                vol = item.get("volumeInfo", {})
                if vol.get("description"):
                    if not description or len(vol["description"]) > len(description):
                        description = vol["description"]
                if vol.get("categories") and not genre:
                    genre = vol["categories"]

            if genre is not None and description is not None:
                logger.info("We've got some results at least")
            batch_df.at[idx, "description"] = description
            batch_df.at[idx, "category"] = genre
            batch_df.at[idx, "enriched"] = True
        return True
    
    def load_dataset(self, dataset_name: str) -> pd.DataFrame:
        if dataset_name != "books":
            raise ValueError("Invalid dataset name: must be 'books'")

        df = self.load_saved_data()
        return df

    async def load_dataset_async(self, dataset_name: str) -> pd.DataFrame:
        if dataset_name != "books":
            raise ValueError("Invalid dataset name: must be 'books'")

        df = self.load_saved_data()
        false_indices = df.index[~df["enriched"]]
        
        if not self.download or len(false_indices) == 0:
            return df

        start_idx = false_indices[0]
        if self.limit is not None:
            residual_dataframe = df.loc[start_idx:start_idx + self.limit - 1]
        else:
            residual_dataframe = df.loc[start_idx:]
        
        async with aiohttp.ClientSession() as session:
            for start in tqdm_asyncio(range(0, len(residual_dataframe), self.batch_size)):
                if start % self.save_rate == 0:
                    self.save(df)
                batch_df = residual_dataframe.iloc[start:start+self.batch_size]
                if not await self.enrich_batch(session, batch_df):
                    logger.info("Dataset enrichment stopped. Aborting... It is due to only empty responses in the batch. If your batch is too small it be accidental. It is needed because of risk of being blocked by Google.")
                    break

        logger.info("Dataset enrichment successfully finished")
        self.save(df)
        return df
