import unittest

from qdrant_client import models

from src.utils.qdrant.db_connection import build_points


class BuildPointsTest(unittest.TestCase):
    def test_metadata_contains_review_and_chunk_info(self) -> None:
        paragraphs = ["First chunk", "Second chunk", "Third chunk"]
        points = build_points(
            review={
                "title": "Ball x Pit review - a laboratory of potential",
                "subtitle": "A laboratory of potential",
                "rating": 4,
                "url": "https://example.test/ball-x-pit-review",
                "web": "eurogamer",
                "metadata": {
                    "developer": "Kenny Sun",
                    "game_name": "Incorrect override",
                    "chunk_count": 99,
                },
            },
            game_name="Ball x Pit",
            paragraphs=paragraphs,
            dense_embeddings=[[0.1] * 384, [0.2] * 384, [0.3] * 384],
            sparse_embeddings=[
                models.SparseVector(indices=[1], values=[0.1]),
                models.SparseVector(indices=[2], values=[0.2]),
                models.SparseVector(indices=[3], values=[0.3]),
            ],
        )

        self.assertEqual(len(points), 3)

        for chunk_index, point in enumerate(points):
            payload = point.payload or {}
            metadata = payload["metadata"]

            self.assertEqual(set(payload), {"review_chunk", "metadata"})
            self.assertNotIn("title", payload)
            self.assertNotIn("game_name", payload)
            self.assertNotIn("rating", payload)
            self.assertNotIn("url", payload)
            self.assertNotIn("web", payload)
            self.assertNotIn("chunk_index", payload)

            self.assertEqual(payload["review_chunk"], paragraphs[chunk_index])
            self.assertEqual(
                metadata["title"], "Ball x Pit review - a laboratory of potential"
            )
            self.assertEqual(metadata["subtitle"], "A laboratory of potential")
            self.assertEqual(metadata["game_name"], "Ball x Pit")
            self.assertEqual(metadata["rating"], 4)
            self.assertEqual(metadata["url"], "https://example.test/ball-x-pit-review")
            self.assertEqual(metadata["web"], "eurogamer")
            self.assertEqual(metadata["chunk_index"], chunk_index)
            self.assertEqual(metadata["chunk_count"], len(paragraphs))
            self.assertEqual(metadata["developer"], "Kenny Sun")


if __name__ == "__main__":
    unittest.main()
