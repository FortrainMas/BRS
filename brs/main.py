import brs.config as config
from brs.ingestion import LoadKaggleDataset, LoadLocalDataset

LoadKaggleDataset().load_dataset("books").head()
