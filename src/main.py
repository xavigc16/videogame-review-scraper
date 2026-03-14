from sentence_transformers import SentenceTransformer

from src.crawlers.eurogamer.eurogamer_crawler import (
    scrape_eurogamer,
)
from src.utils.pgvector.db_connection import conn


def main():
    reviews = scrape_eurogamer()
    model = SentenceTransformer("all-MiniLM-L6-v2")

    for review in reviews:
        for paragraph in review["content"]:
            embedding = model.encode(paragraph).tolist()
            conn.execute(
                "INSERT INTO game_reviews (title, game_name, rating, url, web, review_chunk, embedding) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (
                    review["title"],
                    review["subtitle"].replace(" review", ""),
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
