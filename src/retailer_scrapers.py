"""
Direct scrapers for boutique Blu-ray retailer websites.
Searches retailer sites directly to find special editions that may not appear in Google Shopping.
"""

from __future__ import annotations

import re
import logging
import requests
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, TYPE_CHECKING
from dataclasses import dataclass
from bs4 import BeautifulSoup
from urllib.parse import urljoin, quote_plus

if TYPE_CHECKING:
    from .llm_service import OpenAIService

logger = logging.getLogger(__name__)


@dataclass
class RetailerResult:
    """A product found on a retailer site."""
    title: str
    price: Optional[float]
    url: str
    retailer: str
    edition_type: str  # e.g., "Criterion Collection", "Arrow Video"
    thumbnail: Optional[str] = None
    in_stock: bool = True


class RetailerScraper(ABC):
    """Base class for retailer scrapers."""

    name: str = "Unknown"
    base_url: str = ""
    edition_type: str = "Boutique Release"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        })

    @abstractmethod
    def search(self, movie_title: str, year: Optional[int] = None) -> List[RetailerResult]:
        """Search for a movie on this retailer's site."""
        pass

    def _extract_price(self, price_str: str) -> Optional[float]:
        """Extract numeric price from string."""
        if not price_str:
            return None
        matches = re.findall(r'\d+\.?\d*', price_str.replace(',', ''))
        if matches:
            return float(matches[0])
        return None

    def _normalize_thumbnail(self, url: str) -> Optional[str]:
        """Normalize thumbnail URL."""
        if not url:
            return None
        if url.startswith('//'):
            return 'https:' + url
        return url


class ShopifyScraper(RetailerScraper):
    """
    Generic scraper for Shopify-based retailer sites.
    Configurable via class attributes or __init__ parameters.
    """

    # CSS selectors for Shopify sites (can be overridden)
    product_selectors = '.product-card, .product-item, [data-product-card], a[href*="/products/"]'
    title_selectors = '.product-card__title, .product-title, h3 a, h2 a'
    price_selectors = '.price, .product-price, [data-price], .money'
    max_results = 15

    def search(self, movie_title: str, year: Optional[int] = None) -> List[RetailerResult]:
        results = []
        search_url = f"{self.base_url}/search?q={quote_plus(movie_title)}&type=product"

        try:
            response = self.session.get(search_url, timeout=30)
            response.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"{self.name} search failed: {e}")
            return results

        soup = BeautifulSoup(response.text, 'html.parser')
        seen_urls = set()

        # Try product card selectors first
        products = soup.select(self.product_selectors)

        for product in products[:self.max_results]:
            try:
                result = self._parse_product(product, seen_urls)
                if result:
                    results.append(result)
            except Exception as e:
                logger.debug(f"Error parsing {self.name} product: {e}")
                continue

        logger.info(f"{self.name}: found {len(results)} results for '{movie_title}'")
        return results

    def _parse_product(self, product, seen_urls: set) -> Optional[RetailerResult]:
        """Parse a product element into a RetailerResult."""
        # Get URL first - skip if already seen or invalid
        link = product.select_one('a[href*="/products/"]')
        if not link:
            # Maybe the product itself is the link
            if product.name == 'a' and '/products/' in product.get('href', ''):
                link = product
            else:
                return None

        url = urljoin(self.base_url, link.get('href', ''))
        if url in seen_urls or '/collections/' in url:
            return None
        seen_urls.add(url)

        # Get title
        title_elem = product.select_one(self.title_selectors)
        title = title_elem.get_text(strip=True) if title_elem else None

        # Fallback: try link text or img alt
        if not title or len(title) < 3:
            title = link.get_text(strip=True)
        if not title or len(title) < 3:
            img = product.select_one('img')
            if img:
                title = img.get('alt', '')
        if not title or len(title) < 3:
            return None

        # Get price
        parent = product if product.name in ['div', 'article', 'li'] else product.find_parent(['div', 'article', 'li'])
        price = None
        price_elem = product.select_one(self.price_selectors)
        if not price_elem and parent:
            price_elem = parent.select_one(self.price_selectors)
        if price_elem:
            price = self._extract_price(price_elem.get_text())

        # Get thumbnail
        img = product.select_one('img')
        if not img and parent:
            img = parent.select_one('img')
        thumbnail = self._normalize_thumbnail(img.get('src', '') or img.get('data-src', '')) if img else None

        return RetailerResult(
            title=title,
            price=price,
            url=url,
            retailer=self.name,
            edition_type=self.edition_type,
            thumbnail=thumbnail,
        )


# Shopify-based retailers - minimal configuration needed
class VinegarSyndromeScraper(ShopifyScraper):
    name = "Vinegar Syndrome"
    base_url = "https://vinegarsyndrome.com"
    edition_type = "Vinegar Syndrome"


class SeverinFilmsScraper(ShopifyScraper):
    name = "Severin Films"
    base_url = "https://severinfilms.com"
    edition_type = "Severin Films"


class GrindHouseVideoScraper(ShopifyScraper):
    name = "Grindhouse Video"
    base_url = "https://www.grindhousevideo.com"
    edition_type = "Boutique Release"


class ArrowVideoScraper(RetailerScraper):
    """Scraper for Arrow Video/Arrow Films (custom React/Astro frontend)."""

    name = "Arrow Video"
    base_url = "https://www.arrowfilms.com"
    edition_type = "Arrow Video"

    def search(self, movie_title: str, year: Optional[int] = None) -> List[RetailerResult]:
        results = []
        search_url = f"{self.base_url}/search?q={quote_plus(movie_title)}"

        try:
            response = self.session.get(search_url, timeout=30)
            response.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"Arrow Video search failed: {e}")
            return results

        soup = BeautifulSoup(response.text, 'html.parser')
        product_links = soup.select('a[href*="/product/"]')

        seen_urls = set()
        for link in product_links[:20]:
            try:
                url = urljoin(self.base_url, link.get('href', ''))
                if url in seen_urls:
                    continue
                seen_urls.add(url)

                title = link.get_text(strip=True)
                if not title or len(title) < 3:
                    continue

                # Try to find price and thumbnail nearby
                parent = link.find_parent(['div', 'article', 'li'])
                price = None
                thumbnail = None

                if parent:
                    price_elem = parent.select_one('[class*="price"], .price')
                    if price_elem:
                        price = self._extract_price(price_elem.get_text())

                    img = parent.select_one('img')
                    if img:
                        thumbnail = self._normalize_thumbnail(img.get('src', '') or img.get('data-src', ''))

                results.append(RetailerResult(
                    title=title,
                    price=price,
                    url=url,
                    retailer=self.name,
                    edition_type=self.edition_type,
                    thumbnail=thumbnail,
                ))
            except Exception as e:
                logger.debug(f"Error parsing Arrow Video product: {e}")
                continue

        logger.info(f"Arrow Video: found {len(results)} results for '{movie_title}'")
        return results


class DiabolikDVDScraper(RetailerScraper):
    """Scraper for Diabolik DVD - Magento-based aggregator of boutique releases."""

    name = "Diabolik DVD"
    base_url = "https://www.diabolikdvd.com"
    edition_type = "Boutique Release"

    # Edition detection patterns
    EDITION_PATTERNS = {
        'criterion': "Criterion Collection",
        'arrow': "Arrow Video",
        'shout': "Scream Factory",
        'scream factory': "Scream Factory",
        'kino': "Kino Lorber",
        'vinegar': "Vinegar Syndrome",
        'severin': "Severin Films",
        '88 films': "88 Films",
        'eureka': "Eureka/Masters of Cinema",
        'masters of cinema': "Eureka/Masters of Cinema",
    }

    def search(self, movie_title: str, year: Optional[int] = None) -> List[RetailerResult]:
        results = []
        search_url = f"{self.base_url}/catalogsearch/result/?q={quote_plus(movie_title)}"

        try:
            response = self.session.get(search_url, timeout=30)
            response.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"Diabolik DVD search failed: {e}")
            return results

        soup = BeautifulSoup(response.text, 'html.parser')
        products = soup.select('.product-item, .item.product')

        for product in products[:15]:
            try:
                title_link = product.select_one('.product-item-link, .product-name a')
                if not title_link:
                    continue

                title = title_link.get_text(strip=True)
                url = title_link.get('href', '')
                if not url.startswith('http'):
                    url = urljoin(self.base_url, url)

                # Get price
                price_elem = product.select_one('.price, .regular-price')
                price = self._extract_price(price_elem.get_text()) if price_elem else None

                # Get thumbnail
                img = product.select_one('img.product-image-photo, .product-image img')
                thumbnail = self._normalize_thumbnail(img.get('src', '') or img.get('data-src', '')) if img else None

                # Detect edition type from title
                edition = self._detect_edition(title)

                results.append(RetailerResult(
                    title=title,
                    price=price,
                    url=url,
                    retailer=self.name,
                    edition_type=edition,
                    thumbnail=thumbnail,
                ))
            except Exception as e:
                logger.debug(f"Error parsing Diabolik product: {e}")
                continue

        logger.info(f"Diabolik DVD: found {len(results)} results for '{movie_title}'")
        return results

    def _detect_edition(self, title: str) -> str:
        """Detect boutique label from product title."""
        title_lower = title.lower()
        for pattern, label in self.EDITION_PATTERNS.items():
            if pattern in title_lower:
                return label
        return self.edition_type


class SerpAPISiteSearcher:
    """
    Uses SerpAPI to search specific retailer sites that block direct scraping.
    This uses Google web search with site: operator to find products.
    """

    # Sites that block direct scraping but can be found via Google
    PROTECTED_SITES = {
        "criterion.com": "Criterion Collection",
        "kinolorber.com": "Kino Lorber",
        "shoutfactory.com": "Shout Factory",
        "shop.bfi.org.uk": "BFI",
        "eurekavideo.co.uk": "Eureka/Masters of Cinema",
        "88films.co.uk": "88 Films",
    }

    def __init__(self, api_key: str):
        self.api_key = api_key

    def search(self, movie_title: str, year: Optional[int] = None, alternative_titles: Optional[List[str]] = None) -> List[RetailerResult]:
        """Search protected sites via SerpAPI Google search."""
        from serpapi import GoogleSearch

        results = []

        # Build site: query for all protected sites
        site_query = " OR ".join([f"site:{site}" for site in self.PROTECTED_SITES.keys()])

        # Build title query with alternatives
        title_query = self._build_title_query(movie_title, alternative_titles)
        query = f'{title_query} blu-ray ({site_query})'
        if year:
            query = f'{title_query} {year} blu-ray ({site_query})'

        logger.debug(f"SerpAPI site search query: {query}")

        try:
            params = {
                "api_key": self.api_key,
                "engine": "google",
                "q": query,
                "num": 20,
            }
            search = GoogleSearch(params)
            data = search.get_dict()

            for item in data.get("organic_results", []):
                result = self._parse_result(item)
                if result:
                    results.append(result)

            logger.info(f"SerpAPI site search: found {len(results)} results for '{movie_title}'")

        except Exception as e:
            logger.error(f"SerpAPI site search failed: {e}")

        return results

    def _build_title_query(self, movie_title: str, alternative_titles: Optional[List[str]] = None) -> str:
        """Build search query with title variations."""
        all_titles = [movie_title]
        if alternative_titles:
            for alt in alternative_titles:
                if alt.lower() != movie_title.lower() and alt not in all_titles:
                    all_titles.append(alt)

        if len(all_titles) > 1:
            title_query = " OR ".join([f'"{t}"' for t in all_titles[:4]])
            return f"({title_query})"
        return f'"{movie_title}"'

    def _parse_result(self, item: Dict) -> Optional[RetailerResult]:
        """Parse a SerpAPI result into RetailerResult."""
        try:
            title = item.get("title", "")
            url = item.get("link", "")
            snippet = item.get("snippet", "")

            # Determine retailer from URL
            retailer = "Boutique Retailer"
            edition_type = "Boutique Release"
            for site, label in self.PROTECTED_SITES.items():
                if site in url:
                    retailer = label
                    edition_type = label
                    break

            # Try to extract price from snippet
            price = None
            price_match = re.search(r'\$(\d+\.?\d*)', snippet)
            if price_match:
                price = float(price_match.group(1))

            return RetailerResult(
                title=title,
                price=price,
                url=url,
                retailer=retailer,
                edition_type=edition_type,
                thumbnail=item.get("thumbnail"),
            )
        except Exception as e:
            logger.debug(f"Error parsing SerpAPI result: {e}")
            return None


class RetailerSearcher:
    """Searches across multiple boutique retailers."""

    def __init__(
        self,
        serpapi_key: Optional[str] = None,
        llm_service: Optional[OpenAIService] = None
    ):
        self.scrapers: List[RetailerScraper] = [
            VinegarSyndromeScraper(),
            ArrowVideoScraper(),
            SeverinFilmsScraper(),
            GrindHouseVideoScraper(),
            DiabolikDVDScraper(),
        ]
        self.serpapi_key = serpapi_key
        self.llm_service = llm_service
        self.site_searcher = SerpAPISiteSearcher(serpapi_key) if serpapi_key else None

    def _title_matches(self, product_title: str, search_title: str, alternative_titles: Optional[List[str]] = None) -> bool:
        """
        Check if the product title matches the search title or any alternative.
        Uses strict matching for short/generic titles to avoid false positives.
        """
        product_lower = product_title.lower()

        titles_to_check = [search_title]
        if alternative_titles:
            titles_to_check.extend(alternative_titles)

        for title in titles_to_check:
            title_lower = title.lower()
            title_words = title_lower.split()

            # Skip non-ASCII titles (like Japanese characters)
            if not title.isascii():
                continue

            # Strict matching for short/single-word titles
            # Prevents "House" matching "House of Mortal Sin"
            if len(title_words) == 1 and len(title_lower) <= 10:
                # Title at start followed by delimiter or format keyword
                pattern = rf'^{re.escape(title_lower)}(\s*[\[\(\-:]|\s+blu|\s+4k|\s+dvd|\s*$)'
                if re.search(pattern, product_lower):
                    return True
                # Title in brackets/parens
                pattern2 = rf'[\[\(]{re.escape(title_lower)}[\]\)]'
                if re.search(pattern2, product_lower):
                    return True
                continue

            # Multi-word titles: require all words present
            if len(title_words) > 1:
                if all(word in product_lower for word in title_words):
                    return True
            else:
                # Longer single words: simple containment
                if title_lower in product_lower:
                    return True

        return False

    def _year_matches(self, product_title: str, year: int) -> bool:
        """Check if product title's year matches expected year (within 1 year tolerance)."""
        years_in_title = re.findall(r'\b(19\d{2}|20\d{2})\b', product_title)
        if not years_in_title:
            return True  # No year in title - include it
        return any(abs(int(y) - year) <= 1 for y in years_in_title)

    def _get_retailer_query(
        self,
        movie_title: str,
        retailer_name: str,
        year: Optional[int] = None,
        director: Optional[str] = None
    ) -> str:
        """Get optimized query for a specific retailer using LLM if available."""
        if not self.llm_service:
            return movie_title

        try:
            result = self.llm_service.tailor_query_for_retailer(
                movie_title=movie_title,
                retailer_name=retailer_name,
                year=year,
                director=director,
            )
            if result.query:
                logger.debug(f"LLM retailer query for {retailer_name}: {result.query}")
                return result.query
        except Exception as e:
            logger.warning(f"LLM retailer query failed for {retailer_name}: {e}")

        return movie_title

    def search_all(
        self,
        movie_title: str,
        year: Optional[int] = None,
        max_price: Optional[float] = None,
        alternative_titles: Optional[List[str]] = None,
        director: Optional[str] = None
    ) -> List[RetailerResult]:
        """
        Search all retailers for a movie.

        Args:
            movie_title: Movie title to search for
            year: Optional release year for filtering
            max_price: Optional maximum price filter
            alternative_titles: Optional list of alternative titles to accept
            director: Optional director name for LLM query optimization

        Returns:
            Combined list of results from all retailers
        """
        all_results = []

        # Search direct scrapers with LLM-optimized queries
        for scraper in self.scrapers:
            try:
                # Get optimized query for this retailer
                query = self._get_retailer_query(movie_title, scraper.name, year, director)
                results = scraper.search(query, year)
                all_results.extend(results)
            except Exception as e:
                logger.error(f"Error searching {scraper.name}: {e}")

        # Search protected sites via SerpAPI
        if self.site_searcher:
            try:
                site_results = self.site_searcher.search(movie_title, year, alternative_titles)
                all_results.extend(site_results)
            except Exception as e:
                logger.error(f"Error in SerpAPI site search: {e}")

        # Filter: title match
        all_results = [
            r for r in all_results
            if self._title_matches(r.title, movie_title, alternative_titles)
        ]
        logger.info(f"After title filtering: {len(all_results)} results match '{movie_title}'")

        # Filter: price
        if max_price is not None:
            all_results = [r for r in all_results if r.price is None or r.price <= max_price]

        # Filter: year
        if year:
            all_results = [r for r in all_results if self._year_matches(r.title, year)]

        logger.info(f"Total retailer results for '{movie_title}': {len(all_results)}")
        return all_results


# Convenience function
def search_boutique_retailers(
    movie_title: str,
    year: Optional[int] = None,
    max_price: Optional[float] = None,
    serpapi_key: Optional[str] = None,
    alternative_titles: Optional[List[str]] = None,
    llm_service: Optional[OpenAIService] = None,
    director: Optional[str] = None
) -> List[RetailerResult]:
    """Search all boutique retailers for a movie."""
    searcher = RetailerSearcher(serpapi_key=serpapi_key, llm_service=llm_service)
    return searcher.search_all(movie_title, year, max_price, alternative_titles, director)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = search_boutique_retailers("House", year=1977, max_price=100)
    print(f"\nFound {len(results)} results:\n")
    for r in results:
        print(f"  {r.retailer}: {r.title} - ${r.price} - {r.url}")
