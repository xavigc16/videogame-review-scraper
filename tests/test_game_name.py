import unittest

from src.main import extract_game_name


class ExtractGameNameTest(unittest.TestCase):
    def test_eurogamer_review_dash_subtitle(self) -> None:
        review = {"title": "Ball x Pit review - a laboratory of potential"}

        self.assertEqual(extract_game_name(review), "Ball x Pit")

    def test_eurogamer_early_access_review(self) -> None:
        review = {"title": "Skate Early Access review - four wheels and a dream"}

        self.assertEqual(extract_game_name(review), "Skate")

    def test_ign_spanish_analysis_dash_subtitle(self) -> None:
        review = {"title": "Análisis de Pragmata - Una exquisita e inesperada armonía"}

        self.assertEqual(extract_game_name(review), "Pragmata")

    def test_review_colon_subtitle(self) -> None:
        review = {"title": "Game Name review: subtitle"}

        self.assertEqual(extract_game_name(review), "Game Name")

    def test_title_without_review_marker_is_preserved(self) -> None:
        review = {"title": "A Plain Game Title"}

        self.assertEqual(extract_game_name(review), "A Plain Game Title")

    def test_empty_title_returns_unknown_game(self) -> None:
        review = {"title": "", "subtitle": "Do not use me as a game name"}

        self.assertEqual(extract_game_name(review), "Unknown game")


if __name__ == "__main__":
    unittest.main()
