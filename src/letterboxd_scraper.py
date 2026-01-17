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
    director: Optional[str] = None
    alternative_titles: Optional[List[str]] = None

    def __str__(self) -> str:
        parts = [self.title]
        if self.year:
            parts[0] = f"{self.title} ({self.year})"
        if self.director:
            parts.append(f"dir. {self.director}")
        return " - ".join(parts)

    def get_search_title(self) -> str:
        """
        Get the best title for searching.
        For generic English titles, prefer a more specific alternative title
        (e.g., romanized Japanese title like "Hausu" instead of "House").
        """
        if not self.alternative_titles:
            return self.title

        # Common generic English words that benefit from alternative titles
        generic_words = {'house', 'ring', 'pulse', 'cure', 'audition', 'mother',
                        'father', 'brother', 'sister', 'home', 'dark', 'gate'}

        title_lower = self.title.lower()

        # Check if title is generic (single common word or very short)
        if title_lower in generic_words or len(self.title) <= 5:
            # Look for a romanized alternative (Latin characters, not the same as title)
            for alt in self.alternative_titles:
                # Skip if same as original title
                if alt.lower() == title_lower:
                    continue
                # Prefer romanized titles (ASCII-friendly, good for search)
                if alt.isascii() and len(alt) >= 3:
                    logger.debug(f"Using alternative title '{alt}' instead of '{self.title}'")
                    return alt

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

    def fetch_movie_details(self, movie: Movie) -> Movie:
        """
        Fetch additional details (title, director, year) from the movie's Letterboxd page.
        Returns the same Movie object with updated fields.
        """
        if not movie.letterboxd_url:
            return movie

        try:
            response = self.session.get(movie.letterboxd_url, timeout=30)
            response.raise_for_status()
        except requests.RequestException as e:
            logger.warning(f"Failed to fetch details for {movie.letterboxd_url}: {e}")
            return movie

        soup = BeautifulSoup(response.text, "html.parser")

        # Extract title and year from og:title
        # Format: <meta property="og:title" content="Movie Title (1982)">
        og_title = soup.select_one('meta[property="og:title"]')
        if og_title:
            content = og_title.get("content", "")
            year_match = re.search(r"\((\d{4})\)$", content)
            if year_match:
                # Extract title (everything before the year)
                if not movie.title:
                    movie.title = content[:year_match.start()].strip()
                # Extract year
                if not movie.year:
                    movie.year = int(year_match.group(1))
                    logger.debug(f"Found year for {movie.title}: {movie.year}")
            elif not movie.title:
                # No year in og:title, use whole content as title
                movie.title = content.strip()

        # Extract director from the credits section
        # Format: <a class="contributor" href="/director/...">Director Name</a>
        director_link = soup.select_one('a.contributor[href*="/director/"]')
        if director_link:
            director_name = director_link.get_text(strip=True)
            movie.director = director_name
            logger.debug(f"Found director for {movie.title}: {director_name}")

        # Extract alternative titles
        # Look for the "Alternative Titles" section in the page
        alt_titles = self._extract_alternative_titles(soup)
        if alt_titles:
            movie.alternative_titles = alt_titles
            logger.debug(f"Found {len(alt_titles)} alternative titles for {movie.title}: {alt_titles[:3]}...")

        return movie

    def _extract_alternative_titles(self, soup: BeautifulSoup) -> Optional[List[str]]:
        """Extract alternative titles from a Letterboxd film page."""
        alt_titles = []

        # Method 1: Look for "Alternative Titles" section header
        # The structure is typically: <h3><span>Alternative Titles</span></h3> followed by text
        alt_header = soup.find('h3', string=lambda s: s and 'Alternative' in s)
        if not alt_header:
            # Try finding span inside h3
            for h3 in soup.find_all('h3'):
                span = h3.find('span')
                if span and 'Alternative' in span.get_text():
                    alt_header = h3
                    break

        if alt_header:
            # Get the next sibling or parent's text content
            # Alternative titles are often in a <p> or text node after the header
            next_elem = alt_header.find_next_sibling()
            if next_elem:
                alt_text = next_elem.get_text(strip=True)
                if alt_text:
                    # Split by common delimiters
                    for title in re.split(r'[,،、]', alt_text):
                        title = title.strip()
                        if title and len(title) >= 2:
                            alt_titles.append(title)

        # Method 2: Look in the tab content for alternative titles
        # Letterboxd sometimes puts them in a details tab
        details_section = soup.select_one('.film-details, .tabbed-content')
        if details_section and not alt_titles:
            text = details_section.get_text()
            if 'Alternative' in text:
                # Find the text after "Alternative Titles"
                match = re.search(r'Alternative\s+Titles?\s*[:\s]*([^\n]+)', text, re.IGNORECASE)
                if match:
                    alt_text = match.group(1)
                    for title in re.split(r'[,،、]', alt_text):
                        title = title.strip()
                        if title and len(title) >= 2:
                            alt_titles.append(title)

        # Method 3: Check meta tags for alternate names
        for meta in soup.find_all('meta', attrs={'property': 'og:locale:alternate'}):
            # Sometimes alternative titles are in locale-specific meta tags
            pass  # Letterboxd doesn't use this, but kept for potential future use

        return alt_titles if alt_titles else None

    def enrich_movies(self, movies: List[Movie], delay: float = 0.5) -> List[Movie]:
        """
        Fetch detailed info for all movies in a list.

        Args:
            movies: List of movies to enrich
            delay: Delay between requests to be respectful to Letterboxd

        Returns:
            Same list with enriched movie data
        """
        import time

        total = len(movies)
        logger.info(f"Enriching {total} movies with director info...")

        for i, movie in enumerate(movies, 1):
            logger.info(f"Fetching details {i}/{total}: {movie.title}")
            self.fetch_movie_details(movie)

            # Be respectful to Letterboxd
            if i < total:
                time.sleep(delay)

        enriched_count = sum(1 for m in movies if m.director)
        logger.info(f"Enriched {enriched_count}/{total} movies with director info")

        return movies


def get_movies_from_list(list_url: str, enrich: bool = True) -> List[Movie]:
    """
    Convenience function to scrape movies from a list.

    Args:
        list_url: URL of the Letterboxd list
        enrich: If True, fetch director info for each movie (slower but more accurate searches)

    Returns:
        List of Movie objects
    """
    scraper = LetterboxdScraper()
    movies = scraper.scrape_list(list_url)

    if enrich and movies:
        movies = scraper.enrich_movies(movies)

    return movies


if __name__ == "__main__":
    # Test the scraper
    logging.basicConfig(level=logging.INFO)
    test_url = "https://letterboxd.com/brandt_clawson/list/my-hater-movie-club-list-2026/"
    movies = get_movies_from_list(test_url, enrich=True)

    print(f"\nFound {len(movies)} movies:\n")
    for i, movie in enumerate(movies, 1):
        print(f"{i:3}. {movie}")
