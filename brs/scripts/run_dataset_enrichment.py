import brs.ingestion.google_books_enrichment as google_books_enrichment

if __name__ == "__main__":
    google_books_enrichment.GoogleBooksEnrichment(limit=100).load_dataset("books")
