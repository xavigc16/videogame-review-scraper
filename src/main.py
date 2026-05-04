from sentence_transformers import SentenceTransformer

from src.crawlers.eurogamer.eurogamer_crawler import (
    scrape_eurogamer,
)
from src.utils.pgvector.db_connection import conn


def extract_game_name(review):
    title = review.get("title") or ""
    subtitle = review.get("subtitle") or ""
    source = title or subtitle

    for suffix in (" review", " - review"):
        if source.lower().endswith(suffix):
            return source[: -len(suffix)].strip()

    return source.strip() or "Unknown game"


def main():
    reviews = scrape_eurogamer()
    model = SentenceTransformer("all-MiniLM-L6-v2")

    for review in reviews:
        paragraphs = review["content"]
        embeddings = model.encode(paragraphs).tolist()
        game_name = extract_game_name(review)

        for paragraph, embedding in zip(paragraphs, embeddings, strict=True):
            conn.execute(
                "INSERT INTO game_reviews (title, game_name, rating, url, web, review_chunk, embedding) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (
                    review["title"],
                    game_name,
                    review["rating"],
                    review["url"],
                    review["web"],
                    paragraph,
                    embedding,
                ),
            )


if __name__ == "__main__":
    try:
        main()
    finally:
        conn.close()
