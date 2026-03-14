import psycopg
from pgvector.psycopg import register_vector


conn = psycopg.connect(
    dbname="game_review",
    user="postgres",
    password="mysecretpassword",
    host="localhost",
    autocommit=True,
)

conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
register_vector(conn)

conn.execute("DROP TABLE IF EXISTS game_reviews")
conn.execute("""
    CREATE TABLE IF NOT EXISTS game_reviews (
        id bigserial PRIMARY KEY,
        title text,
        game_name text,
        rating integer,
        url text,
        web text,
        review_chunk text,
        embedding vector(384)
    )
""")  # all-MiniLM-L6-v2 model outputs 384 dimensions.
