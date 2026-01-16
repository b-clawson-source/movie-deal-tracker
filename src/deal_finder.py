"""
Deal finder using SerpAPI to search for physical movie editions.
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

    # Boutique labels for deep search targeted queries
    BOUTIQUE_LABELS = [
        "Criterion",
        "Arrow Video",
        "Vinegar Syndrome",
        "Shout Factory",
        "Kino Lorber",
        "Indicator",
        "Eureka",
        "88 Films",
        "Severin",
        "Blue Underground",
    ]

    def __init__(
        self,
        api_key: str,
        classifier: EditionClassifier,
        max_price: float = 20.0,
        requests_per_minute: int = 30,
        deep_search: bool = False,
    ):
        self.api_key = api_key
        self.classifier = classifier
        self.max_price = max_price
        self.request_delay = 60.0 / requests_per_minute
        self.deep_search = deep_search

    def search_movie(self, movie: Movie) -> List[Deal]:
        """Search for deals on a specific movie."""
        seen_hashes = set()
        deals = []

        # Build search queries (multiple if deep search)
        queries = self._build_queries(movie)

        for query in queries:
            logger.info(f"Searching: {query}")
            try:
                results = self._execute_search(query)
                new_deals = self._process_results(movie, results)

                # Deduplicate
                for deal in new_deals:
                    if deal.deal_hash not in seen_hashes:
                        seen_hashes.add(deal.deal_hash)
                        deals.append(deal)

            except Exception as e:
                logger.error(f"Search failed for {movie.title}: {e}")

            # Rate limiting
            time.sleep(self.request_delay)

        return deals

    def _build_queries(self, movie: Movie) -> List[str]:
        """Build search queries for a movie. Returns multiple queries if deep_search is enabled."""
        title = movie.title
        title_with_year = f"{movie.title} {movie.year}" if movie.year else movie.title

        if not self.deep_search:
            # Standard single query
            return [f'"{title_with_year}" blu-ray OR 4K collector edition']

        # Deep search: multiple targeted queries
        queries = [
            # Broad searches
            f'"{title_with_year}" blu-ray collector edition',
            f'"{title_with_year}" 4K UHD limited edition',
            f'"{title}" steelbook',
            # Title without year (catches more listings)
            f'"{title}" criterion collection blu-ray',
            f'"{title}" arrow video blu-ray',
        ]

        return queries

    def _execute_search(self, query: str) -> Dict[str, Any]:
        """Execute SerpAPI Google Shopping search."""
        num_results = 40 if self.deep_search else 20

        params = {
            "api_key": self.api_key,
            "engine": "google_shopping",
            "q": query,
            "gl": "us",
            "hl": "en",
            "num": num_results,
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

    def find_deals(self, movies: List[Movie]) -> List[Deal]:
        """Search for deals across all movies."""
        all_deals = []
        total = len(movies)

        for i, movie in enumerate(movies, 1):
            logger.info(f"Processing {i}/{total}: {movie.title}")
            deals = self.search_movie(movie)
            all_deals.extend(deals)
            logger.info(f"Found {len(deals)} deals for {movie.title}")

        logger.info(f"Total deals found: {len(all_deals)}")
        return all_deals


if __name__ == "__main__":
    # Test with mock data
    logging.basicConfig(level=logging.INFO)
    print("Deal finder module loaded. Run via main.py for full functionality.")
