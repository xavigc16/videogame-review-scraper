import sys

from loguru import logger


logger.remove()

logger.add(
    sys.stdout,
    colorize=True,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
)

logger.add("scraper_reviews.log", rotation="1 MB", retention="10 days", level="DEBUG")
