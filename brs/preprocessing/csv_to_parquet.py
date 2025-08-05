import os
import logging
from pathlib import Path
from dotenv import load_dotenv

import s3fs
import pandas as pd

from brs.logging_config import setup_logging

setup_logging()
logger = logging.getLogger("CSV to Parquet")

load_dotenv()

STORAGE_OPTIONS = {
    "key": os.getenv("S3_ACCESS_KEY"),
    "secret": os.getenv("S3_SECRET_KEY"),
    "client_kwargs": {
        "endpoint_url": os.getenv("S3_ENDPOINT"),
    },
}


def convert_to_parquet():
    s3 = s3fs.S3FileSystem(**STORAGE_OPTIONS)
    logger.info("S3 is up")

    csv_files = s3.glob("brs/data/raw/*/*.csv")

    for csv_file in csv_files:
        parquet_file = csv_file.replace(".csv", ".parquet")
        if s3.exists(parquet_file):
            logger.info(f"Skipping {csv_file}, Parquet already exists.")
            continue

        csv_uri = f"s3://{csv_file}"
        parquet_uri = f"s3://{parquet_file}"

        logger.info(f"Converting {csv_uri} to {parquet_uri}")
        df = pd.read_csv(csv_uri, storage_options=STORAGE_OPTIONS)
        df = df.astype(str)
        df.to_parquet(parquet_uri, index=False, storage_options=STORAGE_OPTIONS)

        s3.rm(csv_uri)
        logger.info(f"Converted {csv_uri} to {parquet_uri}")
    
    logger.info("S3 is down")
    logger.info("Done converting CSV to Parquet")


if __name__ == "__main__":
    convert_to_parquet()
