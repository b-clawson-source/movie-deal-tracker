"""
Deal finder using SerpAPI to search for physical movie editions.
Includes smart caching that respects sale periods.
"""

import re
import time
import logging
import hashlib
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, asdict
from datetime import datetime

from serpapi import GoogleSearch

from .letterboxd_scraper import Movie
from .edition_classifier import EditionClassifier
from .database import get_db
from .sale_periods import get_cache_ttl_hours, is_sale_period

logger = logging.getLogger(__name__)


@dataclass
class Deal:
    """Represents a found deal."""

    movie_title: str
    product_title: str
    price: float
    retailer: str
    url: str
    similarity_score: float
    matched_example: str
    thumbnail: str = ""
    found_at: str = ""

    def __post_init__(self):
        if not self.found_at:
            self.found_at = datetime.now().isoformat()

    @property
    def deal_hash(self) -> str:
        """Generate unique hash for this deal."""
        key = f"{self.movie_title}|{self.product_title}|{self.retailer}"
        return hashlib.md5(key.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DealFinder:
    """Searches for movie deals using SerpAPI."""

    def __init__(
        self,
        api_key: str,
        classifier: EditionClassifier,
        max_price: float = 20.0,
        requests_per_minute: int = 30,
    ):
        self.api_key = api_key
        self.classifier = classifier
        self.max_price = max_price
        self.request_delay = 60.0 / requests_per_minute

    def search_movie(self, movie: Movie, skip_cache: bool = False) -> List[Deal]:
        """Search for deals on a specific movie.

        Args:
            movie: Movie to search for
            skip_cache: If True, bypass cache and always do fresh search

        Returns:
            List of Deal objects found
        """
        db = get_db()

        # Check sale period status
        is_sale, sale_name = is_sale_period()
        cache_ttl = get_cache_ttl_hours()

        if is_sale:
            logger.info(f"Sale period active ({sale_name}) - using fresh searches")

        # Try cache first (unless skipping or during sale period)
        if not skip_cache and cache_ttl > 0:
            cached = db.get_cached_results(movie.title, self.max_price)
            if cached is not None:
                logger.info(f"Cache hit for '{movie.title}' - returning {len(cached)} cached deals")
                # Convert cached dicts back to Deal objects
                return [Deal(**deal_dict) for deal_dict in cached]

        # Cache miss or skipping cache - do fresh search
        deals = []
        query = self._build_query(movie)
        logger.info(f"Searching: {query}")

        try:
            results = self._execute_search(query)
            deals = self._process_results(movie, results)

            # Cache the results (if caching is enabled)
            if cache_ttl > 0:
                deal_dicts = [deal.to_dict() for deal in deals]
                db.set_cached_results(movie.title, self.max_price, deal_dicts, cache_ttl)
                logger.info(f"Cached {len(deals)} deals for '{movie.title}' (TTL: {cache_ttl}h)")
            else:
                logger.info(f"Skipping cache during sale period")

        except Exception as e:
            logger.error(f"Search failed for {movie.title}: {e}")

        # Rate limiting (only needed for actual API calls)
        time.sleep(self.request_delay)

        return deals

    def _build_query(self, movie: Movie) -> str:
        """Build search query for a movie."""
        title = movie.title
        if movie.year:
            title = f"{movie.title} {movie.year}"

        return f'"{title}" blu-ray OR 4K collector edition'

    def _execute_search(self, query: str) -> Dict[str, Any]:
        """Execute SerpAPI Google Shopping search."""
        params = {
            "api_key": self.api_key,
            "engine": "google_shopping",
            "q": query,
            "gl": "us",
            "hl": "en",
            "num": 20,
        }

        search = GoogleSearch(params)
        return search.get_dict()

    def _process_results(self, movie: Movie, results: Dict) -> List[Deal]:
        """Process search results and filter deals."""
        deals = []

        shopping_results = results.get("shopping_results", [])
        logger.info(f"Found {len(shopping_results)} shopping results")

        for item in shopping_results:
            deal = self._process_item(movie, item)
            if deal:
                deals.append(deal)

        return deals

    def _process_item(self, movie: Movie, item: Dict) -> Optional[Deal]:
        """Process a single shopping result."""
        title = item.get("title", "")
        price_str = item.get("price", "")
        source = item.get("source", "Unknown")
        thumbnail = item.get("thumbnail", "")

        # Get the best available link (product_link preferred, fall back to link)
        link = item.get("product_link") or item.get("link", "")

        # Skip eBay results
        if "ebay" in source.lower():
            return None

        # Extract price
        price = self._extract_price(price_str)
        if price is None:
            logger.debug(f"Could not extract price from: {price_str}")
            return None

        # Check price threshold
        if price > self.max_price:
            logger.debug(f"Price ${price:.2f} exceeds max ${self.max_price:.2f}")
            return None

        # Check if it's a special edition
        is_match, confidence, description = self.classifier.is_special_edition(title)
        if not is_match:
            return None

        return Deal(
            movie_title=movie.title,
            product_title=title,
            price=price,
            retailer=source,
            url=link,
            similarity_score=confidence,
            matched_example=description,
            thumbnail=thumbnail,
        )

    def _extract_price(self, price_str: str) -> Optional[float]:
        """Extract numeric price from string."""
        if not price_str:
            return None

        # Handle various formats: "$19.99", "From $15.00", "$10 - $20"
        # Extract all numbers
        matches = re.findall(r"\d+\.?\d*", price_str)
        if not matches:
            return None

        # Use the lowest price found
        prices = [float(m) for m in matches]
        return min(prices)

    def find_deals(self, movies: List[Movie], skip_cache: bool = False) -> List[Deal]:
        """Search for deals across all movies.

        Args:
            movies: List of movies to search for
            skip_cache: If True, bypass cache for all searches

        Returns:
            List of all Deal objects found
        """
        all_deals = []
        total = len(movies)

        # Log cache status at start
        is_sale, sale_name = is_sale_period()
        cache_ttl = get_cache_ttl_hours()
        if is_sale:
            logger.info(f"Sale period: {sale_name} - caching disabled")
        else:
            logger.info(f"Normal period - cache TTL: {cache_ttl}h")

        for i, movie in enumerate(movies, 1):
            logger.info(f"Processing {i}/{total}: {movie.title}")
            deals = self.search_movie(movie, skip_cache=skip_cache)
            all_deals.extend(deals)
            logger.info(f"Found {len(deals)} deals for {movie.title}")

        logger.info(f"Total deals found: {len(all_deals)}")
        return all_deals


if __name__ == "__main__":
    # Test with mock data
    logging.basicConfig(level=logging.INFO)
    print("Deal finder module loaded. Run via main.py for full functionality.")
