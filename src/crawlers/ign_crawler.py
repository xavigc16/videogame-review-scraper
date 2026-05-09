import time
from collections.abc import Iterable
from urllib.parse import urldefrag, urljoin, urlparse

import requests
from bs4 import BeautifulSoup, Tag

from src.utils.logger_config import logger


BASE_URL = "https://es.ign.com"
REVIEWS_SECTION_URL = "https://es.ign.com/article/review?keyword__type=game"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
}


def normalize_ign_url(url: str) -> str:
    normalized_url = urljoin(BASE_URL, url)
    normalized_url, _fragment = urldefrag(normalized_url)
    return normalized_url


def is_ign_review_url(url: str) -> bool:
    parsed_url = urlparse(url)
    return parsed_url.netloc == "es.ign.com" and "/review/" in parsed_url.path


def get_review_links(session: requests.Session, index_url: str) -> list[str]:
    logger.info(f"Exploring IGN index: {index_url}")

    try:
        response = session.get(index_url, headers=HEADERS, timeout=10)
        response.raise_for_status()
    except requests.RequestException as error:
        logger.exception(f"Failed to fetch IGN index: {error}")
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    links = []
    seen_links = set()
    for a_tag in soup.find_all("a", href=True):
        if not isinstance(a_tag, Tag):
            continue

        href = a_tag.get("href")
        if not isinstance(href, str):
            continue

        normalized_url = normalize_ign_url(href)
        if not is_ign_review_url(normalized_url) or normalized_url in seen_links:
            continue

        links.append(normalized_url)
        seen_links.add(normalized_url)

    logger.success(f"{len(links)} IGN review links found.")
    return links


def extract_meta_content(soup: BeautifulSoup, *selectors: str) -> str | None:
    for selector in selectors:
        meta_tag = soup.select_one(selector)
        if meta_tag and meta_tag.has_attr("content"):
            content = meta_tag.get("content")
            if isinstance(content, str):
                return content.strip()

    return None


def extract_rating(search_roots: Iterable[BeautifulSoup | Tag]) -> int | None:
    review_score = None
    for root in search_roots:
        review_score = root.select_one(".article-review-content .review")
        if review_score:
            break

    if not review_score:
        return None

    rating_text = review_score.get_text(" ", strip=True).split(" ", maxsplit=1)[0]
    try:
        return int(float(rating_text.replace(",", ".")))
    except ValueError:
        logger.warning(f"Failed to convert IGN rating to integer: {rating_text}")
        return None


def scrape_ign_review(url: str, session: requests.Session | None = None) -> dict | None:
    logger.info(f"Initializing IGN scraper for URL: {url}")

    owns_session = session is None
    session = session or requests.Session()

    try:
        response = session.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        article_tag = soup.find("article", class_="article-page")
        article_body = article_tag if isinstance(article_tag, Tag) else soup

        title_tag = article_body.find("h1")
        title = (
            title_tag.get_text(" ", strip=True)
            if isinstance(title_tag, Tag)
            else extract_meta_content(soup, "meta[property='og:title']")
        )
        if not title:
            logger.warning(f"No IGN title found for URL: {url}")

        subtitle_tag = article_body.find("h2")
        subtitle = (
            subtitle_tag.get_text(" ", strip=True)
            if isinstance(subtitle_tag, Tag)
            else "No subtitle"
        )

        rating = extract_rating((article_body, soup))

        paragraphs = []
        for paragraph in article_body.find_all("p"):
            if not isinstance(paragraph, Tag):
                continue

            paragraph_text = paragraph.get_text(" ", strip=True)
            if len(paragraph_text) > 50:
                paragraphs.append(paragraph_text)

        if not paragraphs:
            logger.error(
                "No IGN text extracted. Has the structure of the IGN website changed?"
            )
            return None

        logger.success(f"{len(paragraphs)} IGN paragraphs extracted successfully.")
        return {
            "title": title,
            "subtitle": subtitle,
            "rating": rating,
            "content": paragraphs,
            "url": normalize_ign_url(url),
            "web": "ign",
        }

    except requests.RequestException as error:
        logger.exception(f"Failed to scrape IGN review: {error}")
        return None
    finally:
        if owns_session:
            session.close()


def scrape_ign(max_reviews: int = 1) -> list[dict]:
    with requests.Session() as session:
        review_links = get_review_links(session, REVIEWS_SECTION_URL)[:max_reviews]
        reviews = []

        for index, url in enumerate(review_links, 1):
            logger.info(f"--- Processing IGN review {index}/{len(review_links)} ---")
            review = scrape_ign_review(url, session=session)
            if review:
                reviews.append(review)
                logger.success(f"Saved IGN review in memory for: {review['title']}")

            if index < len(review_links):
                logger.info("Pausing for 3 seconds to avoid overloading the server...")
                time.sleep(3)

    logger.success(f"IGN process finished. Total reviews collected: {len(reviews)}")
    return reviews
