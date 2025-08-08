import os
import logging
import shutil
from pathlib import Path

from dotenv import load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi
import boto3

from brs.logging_config import setup_logging

setup_logging()
logger = logging.getLogger("kaggle_download")

load_dotenv()


def checkup_s3():
    s3 = boto3.client(
        "s3",
        endpoint_url = os.getenv("S3_ENDPOINT"),
        aws_access_key_id=os.getenv("S3_ACCESS_KEY"),
        aws_secret_access_key=os.getenv("S3_SECRET_KEY"),
    )
    s3.close()
    logger.info("S3 is up")

def download_kaggle_dataset():
    checkup_s3()

    api = KaggleApi()
    api.authenticate()

    dataset_name = "arashnic/book-recommendation-dataset"
    local_dir = "/tmp/kaggle-books"
    Path(local_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Downloading from Kaggle")
    api.dataset_download_files(dataset_name, path=local_dir, unzip=True)

    logger.info("Uploading to S3")
    try:
        s3 = boto3.client(
            "s3",
            endpoint_url = os.getenv("S3_ENDPOINT"),
            aws_access_key_id=os.getenv("S3_ACCESS_KEY"),
            aws_secret_access_key=os.getenv("S3_SECRET_KEY"),
        )

        bucket = "brs"
        prefix = "data/raw/kaggle-books"

        for file in Path(local_dir).glob("*csv"):
            key = f"{prefix}/{file.name}"
            s3.upload_file(str(file), bucket, key)

        s3.close()
    except Exception as e:
        logger.error(e)
        logger.error("Failed to upload to S3")

    logger.info("Deleting locally")
    shutil.rmtree(local_dir)

    logger.info("Done")

if __name__ == "__main__":
    download_kaggle_dataset()
