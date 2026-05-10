import os
import uuid
from collections.abc import Iterable
from dataclasses import dataclass

from qdrant_client import QdrantClient, models


POINT_ID_NAMESPACE = uuid.UUID("3caa0a0e-f90d-49b8-8769-c63c872491b8")


@dataclass(frozen=True)
class QdrantSettings:
    url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    api_key: str | None = os.getenv("QDRANT_API_KEY") or None
    collection_name: str = os.getenv("QDRANT_COLLECTION_NAME", "game_reviews")
    dense_vector_name: str = os.getenv("QDRANT_DENSE_VECTOR_NAME", "dense")
    sparse_vector_name: str = os.getenv("QDRANT_SPARSE_VECTOR_NAME", "sparse")
    dense_vector_size: int = int(os.getenv("QDRANT_DENSE_VECTOR_SIZE", "384"))


settings = QdrantSettings()
client = QdrantClient(url=settings.url, api_key=settings.api_key)


def ensure_collection() -> None:
    if client.collection_exists(settings.collection_name):
        return

    client.create_collection(
        collection_name=settings.collection_name,
        vectors_config={
            settings.dense_vector_name: models.VectorParams(
                size=settings.dense_vector_size,
                distance=models.Distance.COSINE,
            )
        },
        sparse_vectors_config={
            settings.sparse_vector_name: models.SparseVectorParams(
                modifier=models.Modifier.IDF,
            )
        },
    )


def build_metadata(
    *,
    review: dict,
    game_name: str,
    chunk_index: int,
    chunk_count: int,
) -> dict:
    review_metadata = review.get("metadata")
    metadata = dict(review_metadata) if isinstance(review_metadata, dict) else {}

    metadata.update(
        {
            "title": review.get("title"),
            "subtitle": review.get("subtitle"),
            "game_name": game_name,
            "rating": review.get("rating"),
            "url": review.get("url"),
            "web": review.get("web"),
            "chunk_index": chunk_index,
            "chunk_count": chunk_count,
        }
    )

    return metadata


def build_payload(
    *,
    review: dict,
    game_name: str,
    paragraph: str,
    chunk_index: int,
    chunk_count: int,
) -> dict:
    return {
        "review_chunk": paragraph,
        "metadata": build_metadata(
            review=review,
            game_name=game_name,
            chunk_index=chunk_index,
            chunk_count=chunk_count,
        ),
    }


def build_points(
    *,
    review: dict,
    game_name: str,
    paragraphs: list[str],
    dense_embeddings: Iterable[list[float]],
    sparse_embeddings: Iterable[models.SparseVector],
) -> list[models.PointStruct]:
    points = []
    chunk_count = len(paragraphs)
    for chunk_index, (paragraph, dense_embedding, sparse_embedding) in enumerate(
        zip(paragraphs, dense_embeddings, sparse_embeddings, strict=True)
    ):
        points.append(
            models.PointStruct(
                id=str(
                    uuid.uuid5(POINT_ID_NAMESPACE, f"{review['url']}#{chunk_index}")
                ),
                vector={
                    settings.dense_vector_name: dense_embedding,
                    settings.sparse_vector_name: sparse_embedding,
                },
                payload=build_payload(
                    review=review,
                    game_name=game_name,
                    paragraph=paragraph,
                    chunk_index=chunk_index,
                    chunk_count=chunk_count,
                ),
            )
        )

    return points
