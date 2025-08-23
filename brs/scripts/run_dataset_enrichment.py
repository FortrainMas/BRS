import asyncio
from brs.ingestion import google_books_enrichment

if __name__ == "__main__":
    enricher = google_books_enrichment.GoogleBooksEnrichment(limit=100000, batch_size=10)
    df = asyncio.run(enricher.load_dataset_async("books"))
    print("Enrichment finished, total rows:", len(df))
