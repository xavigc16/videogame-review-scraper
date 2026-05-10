import argparse
import json
import textwrap
from enum import Enum
from typing import Any

from fastembed import SparseTextEmbedding, TextEmbedding
from pydantic import BaseModel
from qdrant_client import models

from src.main import (
    DENSE_MODEL_NAME,
    FASTEMBED_CACHE_DIR,
    SPARSE_MODEL_NAME,
    configure_model_cache,
)
from src.utils.qdrant.db_connection import client, settings


DEFAULT_LIMIT = 10
CHUNK_PREVIEW_LENGTH = 220


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="qdrant-check",
        description="Inspect the configured Qdrant review collection.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    summary_parser = subparsers.add_parser(
        "summary",
        help="Show collection health, counts, and vector configuration.",
    )
    summary_parser.add_argument(
        "--json",
        action="store_true",
        help="Print summary as JSON.",
    )

    browse_parser = subparsers.add_parser(
        "browse",
        help="Browse stored review chunks without loading vectors.",
    )
    add_result_options(browse_parser)

    search_parser = subparsers.add_parser(
        "search",
        help="Run a hybrid dense+sparse text search against stored review chunks.",
    )
    search_parser.add_argument("query", help="Text query to search for.")
    add_result_options(search_parser)

    return parser.parse_args()


def add_result_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--limit",
        type=positive_int,
        default=DEFAULT_LIMIT,
        help=f"Maximum number of points to return. Default: {DEFAULT_LIMIT}.",
    )
    parser.add_argument(
        "--source",
        help="Filter by review source, for example 'eurogamer' or 'ign'.",
    )
    parser.add_argument(
        "--game",
        help="Filter by exact game_name payload value.",
    )
    parser.add_argument(
        "--rating",
        type=int,
        help="Filter by exact numeric rating.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print results as JSON.",
    )


def positive_int(value: str) -> int:
    parsed_value = int(value)
    if parsed_value < 1:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed_value


def build_filter(args: argparse.Namespace) -> models.Filter | None:
    conditions = []

    if args.source:
        conditions.append(
            models.FieldCondition(
                key="web",
                match=models.MatchValue(value=args.source),
            )
        )

    if args.game:
        conditions.append(
            models.FieldCondition(
                key="game_name",
                match=models.MatchValue(value=args.game),
            )
        )

    if args.rating is not None:
        conditions.append(
            models.FieldCondition(
                key="rating",
                match=models.MatchValue(value=args.rating),
            )
        )

    if not conditions:
        return None

    return models.Filter(must=conditions)


def collection_exists() -> bool:
    return client.collection_exists(settings.collection_name)


def get_summary() -> dict[str, Any]:
    summary: dict[str, Any] = {
        "url": settings.url,
        "collection_name": settings.collection_name,
        "exists": collection_exists(),
    }

    if not summary["exists"]:
        return summary

    collection = client.get_collection(settings.collection_name)
    summary.update(
        {
            "status": enum_value(collection.status),
            "optimizer_status": enum_value(collection.optimizer_status),
            "points_count": collection.points_count,
            "indexed_vectors_count": collection.indexed_vectors_count,
            "segments_count": collection.segments_count,
            "vectors": to_plain(collection.config.params.vectors),
            "sparse_vectors": to_plain(collection.config.params.sparse_vectors),
            "payload_schema": to_plain(collection.payload_schema),
        }
    )
    return summary


def browse(args: argparse.Namespace) -> list[dict[str, Any]]:
    require_collection()
    points, _next_page = client.scroll(
        collection_name=settings.collection_name,
        scroll_filter=build_filter(args),
        limit=args.limit,
        with_payload=True,
        with_vectors=False,
    )
    return [format_point(point) for point in points]


def search(args: argparse.Namespace) -> list[dict[str, Any]]:
    require_collection()
    configure_model_cache()

    dense_model = TextEmbedding(DENSE_MODEL_NAME, cache_dir=str(FASTEMBED_CACHE_DIR))
    sparse_model = SparseTextEmbedding(
        SPARSE_MODEL_NAME, cache_dir=str(FASTEMBED_CACHE_DIR)
    )
    dense_vector = next(dense_model.embed([args.query])).tolist()
    sparse_vector = models.SparseVector(
        **next(sparse_model.embed([args.query])).as_object()
    )

    result = client.query_points(
        collection_name=settings.collection_name,
        prefetch=[
            models.Prefetch(
                query=dense_vector,
                using=settings.dense_vector_name,
                filter=build_filter(args),
                limit=args.limit,
            ),
            models.Prefetch(
                query=sparse_vector,
                using=settings.sparse_vector_name,
                filter=build_filter(args),
                limit=args.limit,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=args.limit,
        with_payload=True,
        with_vectors=False,
    )
    return [format_point(point) for point in result.points]


def require_collection() -> None:
    if not collection_exists():
        raise SystemExit(
            f"Collection '{settings.collection_name}' does not exist at {settings.url}."
        )


def format_point(point: Any) -> dict[str, Any]:
    payload = point.payload or {}
    return {
        "id": str(point.id),
        "score": getattr(point, "score", None),
        "title": payload.get("title"),
        "game_name": payload.get("game_name"),
        "rating": payload.get("rating"),
        "web": payload.get("web"),
        "url": payload.get("url"),
        "metadata": payload.get("metadata"),
        "chunk_index": payload.get("chunk_index"),
        "review_chunk": payload.get("review_chunk"),
    }


def print_summary(summary: dict[str, Any]) -> None:
    print(f"Qdrant URL: {summary['url']}")
    print(f"Collection: {summary['collection_name']}")
    print(f"Exists: {summary['exists']}")

    if not summary["exists"]:
        return

    print(f"Status: {summary['status']}")
    print(f"Optimizer status: {summary['optimizer_status']}")
    print(f"Points: {summary['points_count']}")
    print(f"Indexed vectors: {summary['indexed_vectors_count']}")
    print(f"Segments: {summary['segments_count']}")
    print("Vectors:")
    print_json(summary["vectors"], indent=2, prefix="  ")
    print("Sparse vectors:")
    print_json(summary["sparse_vectors"], indent=2, prefix="  ")
    print("Payload schema:")
    print_json(summary["payload_schema"], indent=2, prefix="  ")


def print_points(points: list[dict[str, Any]]) -> None:
    if not points:
        print("No matching points found.")
        return

    for index, point in enumerate(points, start=1):
        print(f"[{index}] {point['title'] or 'Untitled'}")
        print(f"ID: {point['id']}")
        if point["score"] is not None:
            print(f"Score: {point['score']}")
        print(f"Game: {point['game_name'] or 'Unknown game'}")
        print(f"Rating: {point['rating']}")
        print(f"Source: {point['web']}")
        print(f"URL: {point['url']}")
        print(f"Chunk: {point['chunk_index']}")
        print("Text:")
        print(
            textwrap.fill(
                shorten(point["review_chunk"] or ""),
                width=100,
                initial_indent="  ",
                subsequent_indent="  ",
            )
        )
        if index < len(points):
            print()


def shorten(value: str) -> str:
    value = " ".join(value.split())
    if len(value) <= CHUNK_PREVIEW_LENGTH:
        return value
    return f"{value[: CHUNK_PREVIEW_LENGTH - 3].rstrip()}..."


def print_json(value: Any, *, indent: int = 2, prefix: str = "") -> None:
    rendered = json.dumps(value, ensure_ascii=False, indent=indent)
    if not prefix:
        print(rendered)
        return

    for line in rendered.splitlines():
        print(f"{prefix}{line}")


def to_plain(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {key: to_plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [to_plain(item) for item in value]
    return value


def enum_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    return value


def main() -> None:
    args = parse_args()

    if args.command == "summary":
        summary = get_summary()
        if args.json:
            print_json(summary)
        else:
            print_summary(summary)
        return

    if args.command == "browse":
        points = browse(args)
    else:
        points = search(args)

    if args.json:
        print_json(points)
    else:
        print_points(points)


if __name__ == "__main__":
    main()
