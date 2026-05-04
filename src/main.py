from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import models

from src.crawlers.eurogamer.eurogamer_crawler import (
    scrape_eurogamer,
)
from src.utils.qdrant.db_connection import (
    build_points,
    client,
    ensure_collection,
    settings,
)


DENSE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SPARSE_MODEL_NAME = "Qdrant/bm25"


def extract_game_name(review):
    title = review.get("title") or ""
    subtitle = review.get("subtitle") or ""
    source = title or subtitle

    for suffix in (" review", " - review"):
        if source.lower().endswith(suffix):
            return source[: -len(suffix)].strip()

    return source.strip() or "Unknown game"


def main():
    ensure_collection()
    reviews = scrape_eurogamer()
    dense_model = TextEmbedding(DENSE_MODEL_NAME)
    sparse_model = SparseTextEmbedding(SPARSE_MODEL_NAME)

    for review in reviews:
        paragraphs = review["content"]
        dense_embeddings = [
            embedding.tolist() for embedding in dense_model.embed(paragraphs)
        ]
        sparse_embeddings = [
            models.SparseVector(**sparse_embedding.as_object())
            for sparse_embedding in sparse_model.embed(paragraphs)
        ]
        game_name = extract_game_name(review)

        client.upsert(
            collection_name=settings.collection_name,
            points=build_points(
                review=review,
                game_name=game_name,
                paragraphs=paragraphs,
                dense_embeddings=dense_embeddings,
                sparse_embeddings=sparse_embeddings,
            ),
        )


if __name__ == "__main__":
    main()
