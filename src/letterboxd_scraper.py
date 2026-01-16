"""
Letterboxd list scraper - extracts movie titles from public lists.
"""

import re
import logging
import requests
from bs4 import BeautifulSoup
from typing import List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Movie:
    """Represents a movie from a Letterboxd list."""
    title: str
    year: Optional[int] = None
    letterboxd_url: Optional[str] = None

    def __str__(self) -> str:
        if self.year:
            return f"{self.title} ({self.year})"
        return self.title


class LetterboxdScraper:
    """Scrapes movie titles from Letterboxd lists."""

    BASE_URL = "https://letterboxd.com"
    USER_AGENT = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.USER_AGENT})

    def scrape_list(self, list_url: str) -> List[Movie]:
        """
        Scrape all movies from a Letterboxd list.
        Uses the /detail/ endpoint which renders without JS.
        Handles pagination automatically.
        """
        movies = []
        page = 1

        while True:
            page_url = self._get_page_url(list_url, page)
            logger.info(f"Scraping page {page}: {page_url}")

            page_movies = self._scrape_page(page_url)

            if not page_movies:
                break

            movies.extend(page_movies)
            page += 1

            # Letterboxd lists paginate at 100 items
            if len(page_movies) < 100:
                break

        logger.info(f"Found {len(movies)} movies in list")
        return movies

    def _get_page_url(self, list_url: str, page: int) -> str:
        """Generate paginated URL using the /detail/ endpoint."""
        # Remove trailing slash and ensure we use /detail/ endpoint
        list_url = list_url.rstrip("/")

        # Remove /detail if already present
        if list_url.endswith("/detail"):
            list_url = list_url[:-7]

        if page == 1:
            return f"{list_url}/detail/"
        return f"{list_url}/detail/page/{page}/"

    def _scrape_page(self, url: str) -> List[Movie]:
        """Scrape a single page of the list."""
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return []

        soup = BeautifulSoup(response.text, "html.parser")
        movies = []
        seen_urls = set()

        # Find all film links in the detail view
        # The detail view has links like /film/movie-slug/
        film_links = soup.select('a[href*="/film/"]')

        for link in film_links:
            href = link.get("href", "")

            # Only process actual film links (not user reviews etc)
            if not re.match(r"^/film/[^/]+/?$", href):
                continue

            # Skip duplicates (each film may appear multiple times)
            if href in seen_urls:
                continue
            seen_urls.add(href)

            movie = self._parse_film_link(link, href)
            if movie:
                movies.append(movie)

        return movies

    def _parse_film_link(self, link, href: str) -> Optional[Movie]:
        """Parse movie info from a film link."""
        try:
            # Get title from link text
            title = link.get_text(strip=True)

            if not title:
                return None

            # Try to extract year from title if present (e.g., "Movie Title (2024)")
            year_match = re.search(r"\((\d{4})\)$", title)
            year = None
            if year_match:
                year = int(year_match.group(1))
                title = title[: year_match.start()].strip()

            # Build full Letterboxd URL
            letterboxd_url = f"{self.BASE_URL}{href}"

            return Movie(title=title, year=year, letterboxd_url=letterboxd_url)

        except Exception as e:
            logger.warning(f"Failed to parse movie: {e}")
            return None


def get_movies_from_list(list_url: str) -> List[Movie]:
    """Convenience function to scrape movies from a list."""
    scraper = LetterboxdScraper()
    return scraper.scrape_list(list_url)


if __name__ == "__main__":
    # Test the scraper
    logging.basicConfig(level=logging.INFO)
    test_url = "https://letterboxd.com/brandt_clawson/list/my-hater-movie-club-list-2026/"
    movies = get_movies_from_list(test_url)

    print(f"\nFound {len(movies)} movies:\n")
    for i, movie in enumerate(movies, 1):
        print(f"{i:3}. {movie}")
