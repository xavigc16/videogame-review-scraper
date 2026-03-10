import requests
from bs4 import BeautifulSoup
import time
from utils.logger_config import logger


def scrape_eurogamer_review(url):
    logger.info(f"Initializing scraper for URL: {url}")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9,es;q=0.8",
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code == 200:
            logger.success("Connection successful. Analyzing HTML...")
            soup = BeautifulSoup(response.text, "html.parser")

            page_text = soup.get_text().lower()

            has_game_details = "developer:" in page_text and "publisher:" in page_text
            details_box = soup.find(class_="game-details") or soup.find(
                class_="info-box"
            )

            if not has_game_details and not details_box:
                logger.warning(
                    f"Discarded: No video game details found (Developer/Publisher). Likely hardware -> {url}"
                )
                return None

            title_tag = soup.find("h1")
            title = title_tag.get_text(strip=True) if title_tag else None
            if not title:
                logger.warning("No title found (missing <h1> tag).")
            else:
                logger.info(f"Title extracted: {title}")

            subtitle_tag = soup.find("h2") or soup.select_one(".strapline, .subtitle")
            subtitle = (
                subtitle_tag.get_text(strip=True) if subtitle_tag else "No subtitle"
            )

            article_body = soup.find("article") or soup
            paragraphs = article_body.find_all("p")

            valid_paragraphs = []
            for p in paragraphs:
                text = p.get_text(strip=True)
                if len(text) > 50:
                    valid_paragraphs.append(text)

            if not valid_paragraphs:
                logger.error(
                    "No text extracted. Has the structure of the Eurogamer website changed?"
                )
                return None

            logger.success(
                f"{len(valid_paragraphs)} paragraphs extracted successfully."
            )

            return {
                "title": title,
                "subtitle": subtitle,
                "content": valid_paragraphs,
                "url": url,
            }

        elif response.status_code == 403:
            logger.error("Error 403: Access denied. The antibot system has blocked us.")
        elif response.status_code == 404:
            logger.error("Error 404: The page does not exist. Check the URL.")
        else:
            logger.error(
                f"Failed to connect. Unexpected HTTP code: {response.status_code}"
            )

    except requests.exceptions.Timeout:
        logger.error(
            "The request has taken too long (Timeout). The website might be overloaded."
        )
    except requests.exceptions.ConnectionError:
        logger.error(
            "Network error. Please check your internet connection or if the website is down."
        )
    except Exception as e:
        logger.exception(f"A fatal and uncontrolled error has occurred: {e}")

    return None


def get_links_multiple_pages(base_url, pages_to_scrape=3):
    """
    Navigates through the Eurogamer index jumping from page to page.
    """
    all_links = set()  # We use a set to avoid duplicate links

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    for page in range(1, pages_to_scrape + 1):
        if page == 1:
            current_url = base_url
        else:
            current_url = f"{base_url}?page={page}"

        logger.info(f"🔍 Exploring index: {current_url}")

        try:
            response = requests.get(current_url, headers=headers, timeout=10)

            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")

                for a_tag in soup.find_all("a", href=True):
                    link = a_tag["href"]

                    if link.startswith("/"):
                        link = "https://www.eurogamer.net" + link

                    # URL filter and hardware blacklist
                    if "eurogamer.net" in link and "review" in link and len(link) > 40:
                        forbidden_words = [
                            "digitalfoundry",
                            "hardware",
                            "tech",
                            "mouse",
                            "keyboard",
                            "headset",
                            "monitor",
                            "gpu",
                            "cpu",
                            "view=comments",
                            "auth",
                            "page=",
                        ]
                        is_technology = any(p in link.lower() for p in forbidden_words)

                        if not is_technology:
                            all_links.add(link)

                logger.success(
                    f"Page {page} completed. Accumulated links: {len(all_links)}"
                )

            else:
                logger.error(f"Error {response.status_code} when accessing page {page}")

        except Exception as e:
            logger.exception(f"Fatal error on page {page}: {e}")

        time.sleep(2)

    return list(all_links)


if __name__ == "__main__":
    reviews_section_url = "https://www.eurogamer.net/reviews"

    url_list = get_links_multiple_pages(reviews_section_url, 3)

    test_limit = 18
    urls_to_scrape = url_list[:test_limit]

    all_reviews = []

    for i, url in enumerate(urls_to_scrape, 1):
        logger.info(f"--- Processing review {i}/{len(urls_to_scrape)} ---")

        review_data = scrape_eurogamer_review(url)

        if review_data:
            all_reviews.append(review_data)
            logger.success(f"Saved review in memory for: {review_data['title']}")

        logger.info("Pausing for 3 seconds to avoid overloading the server...")
        time.sleep(3)

    logger.success(f"Process finished. Total reviews collected: {len(all_reviews)}")
