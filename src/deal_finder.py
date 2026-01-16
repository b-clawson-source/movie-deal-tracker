"""
Deal finder using SerpAPI to search for physical movie editions.
"""

import re
import time
import logging
import hashlib
import json
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

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

    def search_movie(self, movie: Movie) -> List[Deal]:
        """Search for deals on a specific movie."""
        deals = []

        # Build search query
        query = self._build_query(movie)
        logger.info(f"Searching: {query}")

        try:
            results = self._execute_search(query)
            deals = self._process_results(movie, results)
        except Exception as e:
            logger.error(f"Search failed for {movie.title}: {e}")

        # Rate limiting
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
        link = item.get("link", "")
        thumbnail = item.get("thumbnail", "")

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


class DealTracker:
    """Tracks found deals to avoid duplicate notifications."""

    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.data_path.parent.mkdir(parents=True, exist_ok=True)
        self.seen_deals: Dict[str, Dict] = {}
        self._load()

    def _load(self):
        """Load previously seen deals."""
        if self.data_path.exists():
            try:
                with open(self.data_path, "r") as f:
                    self.seen_deals = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load deal history: {e}")
                self.seen_deals = {}

    def _save(self):
        """Save seen deals to file."""
        with open(self.data_path, "w") as f:
            json.dump(self.seen_deals, f, indent=2)

    def is_new_deal(self, deal: Deal, days_threshold: int = 7) -> bool:
        """Check if deal is new (not seen in last N days)."""
        deal_hash = deal.deal_hash

        if deal_hash not in self.seen_deals:
            return True

        # Check if enough time has passed
        last_seen = self.seen_deals[deal_hash].get("last_seen", "")
        if last_seen:
            try:
                last_date = datetime.fromisoformat(last_seen)
                days_ago = (datetime.now() - last_date).days
                return days_ago >= days_threshold
            except ValueError:
                return True

        return True

    def mark_seen(self, deal: Deal):
        """Mark a deal as seen."""
        self.seen_deals[deal.deal_hash] = {
            "deal": deal.to_dict(),
            "last_seen": datetime.now().isoformat(),
        }
        self._save()

    def filter_new_deals(
        self, deals: List[Deal], days_threshold: int = 7
    ) -> List[Deal]:
        """Filter to only new deals and mark them as seen."""
        new_deals = []
        for deal in deals:
            if self.is_new_deal(deal, days_threshold):
                new_deals.append(deal)
                self.mark_seen(deal)
        return new_deals


if __name__ == "__main__":
    # Test with mock data
    logging.basicConfig(level=logging.INFO)
    print("Deal finder module loaded. Run via main.py for full functionality.")
