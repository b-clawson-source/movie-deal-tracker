"""
OpenAI API integration for LLM-enhanced classification and search refinement.
Uses a fallback-only approach for cost efficiency.
"""

import hashlib
import logging
from dataclasses import dataclass
from typing import List, Optional

from openai import OpenAI

logger = logging.getLogger(__name__)

# Model selection for cost optimization
MINI_MODEL = "gpt-4.1-mini"  # Cheap, fast - for classification
FULL_MODEL = "gpt-4o"  # Smarter - for search refinement


@dataclass
class LLMClassificationResult:
    """Result from LLM edition classification."""
    is_special: bool
    confidence: float
    label: Optional[str]
    edition_type: Optional[str]
    reason: str


@dataclass
class SearchRefinementResult:
    """Result from LLM search refinement suggestions."""
    alternative_queries: List[str]
    reasoning: str


@dataclass
class QueryExpansionResult:
    """Result from proactive query expansion."""
    queries: List[str]
    reasoning: str


@dataclass
class TitleValidationResult:
    """Result from title validation."""
    is_match: bool
    confidence: float
    reason: str


@dataclass
class BundleDetectionResult:
    """Result from collection/bundle detection."""
    bundle_queries: List[str]
    reasoning: str


@dataclass
class RetailerQueryResult:
    """Result from retailer-specific query tuning."""
    query: str
    reasoning: str


class OpenAIService:
    """OpenAI API integration for movie deal classification and search refinement."""

    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self._cache: dict = {}  # In-memory cache for LLM responses

    def _get_cache_key(self, *args) -> str:
        """Generate cache key from arguments."""
        key_str = "|".join(str(arg) for arg in args)
        return hashlib.md5(key_str.encode()).hexdigest()

    def classify_edition(self, product_title: str, movie_title: str) -> LLMClassificationResult:
        """
        Classify product using GPT-4o-mini (cheap, fast).

        Args:
            product_title: The full product listing title
            movie_title: The movie title we're searching for

        Returns:
            LLMClassificationResult with classification details
        """
        cache_key = self._get_cache_key("classify", product_title, movie_title)
        if cache_key in self._cache:
            logger.debug(f"Cache hit for classification: {product_title[:50]}...")
            return self._cache[cache_key]

        prompt = f"""Analyze this product listing and determine if it's a special/collector's edition of the movie "{movie_title}".

Product title: {product_title}

Respond in this exact format:
IS_SPECIAL: yes or no
CONFIDENCE: 0.0 to 1.0
LABEL: boutique label name if identified (e.g., "Criterion Collection", "Arrow Video") or "none"
EDITION_TYPE: type of edition (e.g., "Steelbook", "Limited Edition", "Collector's Edition") or "Standard"
REASON: brief explanation (one sentence)

Consider these factors:
- Boutique labels (Criterion, Arrow, Shout Factory, Vinegar Syndrome, Kino Lorber, etc.) = special edition
- Keywords like "steelbook", "collector", "limited", "restored", "remastered" = special edition
- Standard retail releases from major studios without special features = NOT special
- DVD format = NOT special (we only want Blu-ray/4K)"""

        try:
            response = self.client.chat.completions.create(
                model=MINI_MODEL,
                max_tokens=256,
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = response.choices[0].message.content
            result = self._parse_classification_response(response_text)
            self._cache[cache_key] = result
            logger.debug(f"LLM classified '{product_title[:50]}...' as special={result.is_special}")
            return result

        except Exception as e:
            logger.error(f"LLM classification failed: {e}")
            raise

    def _parse_classification_response(self, response: str) -> LLMClassificationResult:
        """Parse the LLM response into a structured result."""
        lines = response.strip().split("\n")
        result = {
            "is_special": False,
            "confidence": 0.5,
            "label": None,
            "edition_type": None,
            "reason": "Could not parse LLM response"
        }

        for line in lines:
            line = line.strip()
            if line.startswith("IS_SPECIAL:"):
                value = line.split(":", 1)[1].strip().lower()
                result["is_special"] = value == "yes"
            elif line.startswith("CONFIDENCE:"):
                try:
                    result["confidence"] = float(line.split(":", 1)[1].strip())
                except ValueError:
                    pass
            elif line.startswith("LABEL:"):
                value = line.split(":", 1)[1].strip()
                result["label"] = None if value.lower() == "none" else value
            elif line.startswith("EDITION_TYPE:"):
                value = line.split(":", 1)[1].strip()
                result["edition_type"] = None if value.lower() == "standard" else value
            elif line.startswith("REASON:"):
                result["reason"] = line.split(":", 1)[1].strip()

        return LLMClassificationResult(**result)

    def suggest_search_refinements(
        self,
        movie_title: str,
        year: Optional[int],
        current_results: int,
        original_query: str
    ) -> SearchRefinementResult:
        """
        Suggest alternative search queries using GPT-4o (smarter).

        Args:
            movie_title: The movie title
            year: Release year (if known)
            current_results: Number of results from original query
            original_query: The original search query used

        Returns:
            SearchRefinementResult with alternative queries
        """
        cache_key = self._get_cache_key("refine", movie_title, year, original_query)
        if cache_key in self._cache:
            logger.debug(f"Cache hit for search refinement: {movie_title}")
            return self._cache[cache_key]

        year_info = f" ({year})" if year else ""
        prompt = f"""I'm searching for special/collector's edition Blu-ray or 4K releases of the movie "{movie_title}"{year_info}.

My original search query was: {original_query}
This returned only {current_results} result(s), which is too few.

Suggest 2-3 alternative search queries that might find more results. Consider:
- Alternative titles (international titles, sequel numbering differences)
- Common misspellings or variations
- Removing overly specific terms
- Different boutique label names
- The movie might be part of a box set or collection

Respond in this exact format:
QUERY1: your first alternative query
QUERY2: your second alternative query
QUERY3: your third alternative query (optional)
REASONING: brief explanation of your suggestions

Keep queries focused on finding physical media (Blu-ray, 4K UHD) special editions."""

        try:
            response = self.client.chat.completions.create(
                model=FULL_MODEL,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = response.choices[0].message.content
            result = self._parse_refinement_response(response_text)
            self._cache[cache_key] = result
            logger.info(f"LLM suggested {len(result.alternative_queries)} alternative queries for '{movie_title}'")
            return result

        except Exception as e:
            logger.error(f"LLM search refinement failed: {e}")
            raise

    def _parse_refinement_response(self, response: str) -> SearchRefinementResult:
        """Parse the LLM response into search refinement result."""
        lines = response.strip().split("\n")
        queries = []
        reasoning = "LLM suggested alternative queries"

        for line in lines:
            line = line.strip()
            if line.startswith("QUERY") and ":" in line:
                query = line.split(":", 1)[1].strip()
                if query:
                    queries.append(query)
            elif line.startswith("REASONING:"):
                reasoning = line.split(":", 1)[1].strip()

        return SearchRefinementResult(
            alternative_queries=queries,
            reasoning=reasoning
        )

    def generate_search_queries(
        self,
        movie_title: str,
        year: Optional[int] = None,
        director: Optional[str] = None,
        alternative_titles: Optional[List[str]] = None,
    ) -> QueryExpansionResult:
        """
        Proactively generate multiple search queries for a movie.

        Uses LLM knowledge to create varied queries based on:
        - Alternative/international titles
        - Director collections and box sets
        - Known boutique label affinities
        - Common naming variations

        Args:
            movie_title: Primary movie title
            year: Release year (if known)
            director: Director name (if known)
            alternative_titles: Known alternative titles

        Returns:
            QueryExpansionResult with multiple search queries
        """
        cache_key = self._get_cache_key("expand", movie_title, year, director)
        if cache_key in self._cache:
            logger.debug(f"Cache hit for query expansion: {movie_title}")
            return self._cache[cache_key]

        # Build context for the prompt
        year_info = f"Year: {year}" if year else "Year: unknown"
        director_info = f"Director: {director}" if director else "Director: unknown"
        alt_titles_info = f"Known alternative titles: {', '.join(alternative_titles)}" if alternative_titles else "No known alternative titles"

        prompt = f"""Generate search queries to find special/collector's edition Blu-ray and 4K UHD releases of this movie:

Movie: {movie_title}
{year_info}
{director_info}
{alt_titles_info}

Generate 3-4 different Google Shopping search queries that will maximize chances of finding boutique label releases (Criterion, Arrow, Shout Factory, Vinegar Syndrome, Kino Lorber, etc.).

Consider:
1. The primary title with format keywords (blu-ray, 4K)
2. Alternative/international titles if it's a foreign film
3. Director's name + collection (e.g., "Kubrick Collection blu-ray")
4. Franchise or anthology names if applicable
5. Specific boutique labels known for this type of film

Respond in this exact format:
QUERY1: your first search query
QUERY2: your second search query
QUERY3: your third search query
QUERY4: your fourth search query (optional)
REASONING: brief explanation of your query strategy

Keep queries concise and focused on finding physical media special editions."""

        try:
            response = self.client.chat.completions.create(
                model=MINI_MODEL,  # Use cheaper model for query generation
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = response.choices[0].message.content
            result = self._parse_query_expansion_response(response_text)
            self._cache[cache_key] = result
            logger.info(f"LLM generated {len(result.queries)} search queries for '{movie_title}'")
            return result

        except Exception as e:
            logger.error(f"LLM query expansion failed: {e}")
            raise

    def _parse_query_expansion_response(self, response: str) -> QueryExpansionResult:
        """Parse the LLM response into query expansion result."""
        lines = response.strip().split("\n")
        queries = []
        reasoning = "LLM generated search queries"

        for line in lines:
            line = line.strip()
            if line.startswith("QUERY") and ":" in line:
                query = line.split(":", 1)[1].strip()
                if query:
                    queries.append(query)
            elif line.startswith("REASONING:"):
                reasoning = line.split(":", 1)[1].strip()

        return QueryExpansionResult(
            queries=queries,
            reasoning=reasoning
        )

    def validate_movie_match(
        self,
        product_title: str,
        movie_title: str,
        year: Optional[int] = None,
        director: Optional[str] = None,
        alternative_titles: Optional[List[str]] = None,
    ) -> TitleValidationResult:
        """
        Validate that a product listing is for the correct movie.

        Uses LLM to semantically verify the match, handling:
        - Generic titles (House 1977 vs House 1986)
        - Remakes with same name
        - Similar/confusing titles
        - International title variations

        Args:
            product_title: The product listing title
            movie_title: The movie we're searching for
            year: Release year of the target movie
            director: Director of the target movie
            alternative_titles: Known alternative titles

        Returns:
            TitleValidationResult with match status and confidence
        """
        cache_key = self._get_cache_key("validate", product_title, movie_title, year)
        if cache_key in self._cache:
            logger.debug(f"Cache hit for title validation: {product_title[:40]}...")
            return self._cache[cache_key]

        # Build context
        year_info = f"Year: {year}" if year else "Year: unknown"
        director_info = f"Director: {director}" if director else "Director: unknown"
        alt_info = f"Alternative titles: {', '.join(alternative_titles)}" if alternative_titles else ""

        prompt = f"""Determine if this product listing is for the correct movie.

TARGET MOVIE:
Title: {movie_title}
{year_info}
{director_info}
{alt_info}

PRODUCT LISTING:
{product_title}

Is this product listing for the target movie? Consider:
- There may be multiple movies with the same/similar title (remakes, different countries)
- The product may be for a different year's version
- The product may be for a similarly-named but different film
- Foreign films may be listed under original or translated titles

Respond in this exact format:
IS_MATCH: yes or no
CONFIDENCE: 0.0 to 1.0
REASON: brief explanation (one sentence)"""

        try:
            response = self.client.chat.completions.create(
                model=MINI_MODEL,
                max_tokens=256,
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = response.choices[0].message.content
            result = self._parse_validation_response(response_text)
            self._cache[cache_key] = result
            logger.debug(f"Title validation for '{product_title[:40]}...': match={result.is_match}")
            return result

        except Exception as e:
            logger.error(f"LLM title validation failed: {e}")
            raise

    def _parse_validation_response(self, response: str) -> TitleValidationResult:
        """Parse the LLM response into validation result."""
        lines = response.strip().split("\n")
        result = {
            "is_match": True,  # Default to match (benefit of doubt)
            "confidence": 0.5,
            "reason": "Could not parse LLM response"
        }

        for line in lines:
            line = line.strip()
            if line.startswith("IS_MATCH:"):
                value = line.split(":", 1)[1].strip().lower()
                result["is_match"] = value == "yes"
            elif line.startswith("CONFIDENCE:"):
                try:
                    result["confidence"] = float(line.split(":", 1)[1].strip())
                except ValueError:
                    pass
            elif line.startswith("REASON:"):
                result["reason"] = line.split(":", 1)[1].strip()

        return TitleValidationResult(**result)

    def detect_bundles(
        self,
        movie_title: str,
        year: Optional[int] = None,
        director: Optional[str] = None,
    ) -> BundleDetectionResult:
        """
        Detect potential box sets or collections containing this movie.

        Uses LLM knowledge to identify:
        - Director filmography collections
        - Studio anthology sets
        - Franchise box sets
        - Thematic collections (horror, sci-fi classics, etc.)

        Args:
            movie_title: The movie title
            year: Release year (if known)
            director: Director name (if known)

        Returns:
            BundleDetectionResult with search queries for potential bundles
        """
        cache_key = self._get_cache_key("bundle", movie_title, year, director)
        if cache_key in self._cache:
            logger.debug(f"Cache hit for bundle detection: {movie_title}")
            return self._cache[cache_key]

        year_info = f" ({year})" if year else ""
        director_info = f"Director: {director}" if director else "Director: unknown"

        prompt = f"""Identify potential box sets, collections, or anthologies that might contain the movie "{movie_title}"{year_info}.

{director_info}

Think about:
1. Director filmography collections (e.g., "Kubrick Collection", "Kurosawa Box Set")
2. Franchise/series box sets (e.g., "Alien Anthology", "Star Wars Saga")
3. Studio collections (e.g., "Universal Monsters", "Hammer Horror Collection")
4. Thematic collections (e.g., "80s Horror Classics", "Japanese New Wave")
5. Boutique label collections (e.g., "Criterion Eclipse Series", "Arrow Academy")

Generate 2-3 search queries specifically for finding this movie in a collection/box set.
Only suggest collections that likely exist and would include this specific film.

Respond in this exact format:
BUNDLE1: search query for first collection type
BUNDLE2: search query for second collection type
BUNDLE3: search query for third collection type (optional)
REASONING: brief explanation of why these collections might exist

If no likely bundles exist, respond with:
BUNDLE1: none
REASONING: explanation of why bundles are unlikely"""

        try:
            response = self.client.chat.completions.create(
                model=MINI_MODEL,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = response.choices[0].message.content
            result = self._parse_bundle_response(response_text)
            self._cache[cache_key] = result
            logger.info(f"Bundle detection for '{movie_title}': {len(result.bundle_queries)} potential bundles")
            return result

        except Exception as e:
            logger.error(f"LLM bundle detection failed: {e}")
            raise

    def _parse_bundle_response(self, response: str) -> BundleDetectionResult:
        """Parse the LLM response into bundle detection result."""
        lines = response.strip().split("\n")
        queries = []
        reasoning = "LLM suggested bundle searches"

        for line in lines:
            line = line.strip()
            if line.startswith("BUNDLE") and ":" in line:
                query = line.split(":", 1)[1].strip()
                # Skip "none" responses
                if query and query.lower() != "none":
                    queries.append(query)
            elif line.startswith("REASONING:"):
                reasoning = line.split(":", 1)[1].strip()

        return BundleDetectionResult(
            bundle_queries=queries,
            reasoning=reasoning
        )

    def tailor_query_for_retailer(
        self,
        movie_title: str,
        retailer_name: str,
        year: Optional[int] = None,
        director: Optional[str] = None,
    ) -> RetailerQueryResult:
        """
        Generate an optimized search query for a specific retailer.

        Different retailers have different naming conventions:
        - Criterion: Uses spine numbers, specific naming
        - Arrow Video: UK-focused, may use different titles
        - Amazon: Long descriptive titles
        - Boutique sites: May use catalog numbers

        Args:
            movie_title: The movie title
            retailer_name: Name of the retailer (e.g., "Criterion", "Arrow Video")
            year: Release year (if known)
            director: Director name (if known)

        Returns:
            RetailerQueryResult with optimized query for the retailer
        """
        cache_key = self._get_cache_key("retailer", movie_title, retailer_name, year)
        if cache_key in self._cache:
            logger.debug(f"Cache hit for retailer query: {movie_title} @ {retailer_name}")
            return self._cache[cache_key]

        year_info = f" ({year})" if year else ""
        director_info = f"Director: {director}" if director else ""

        prompt = f"""Generate an optimized search query for finding "{movie_title}"{year_info} on the {retailer_name} website.

{director_info}

Consider the retailer's typical naming conventions:
- Criterion Collection: Often uses exact titles, spine numbers, or director names
- Arrow Video/Arrow Academy: UK releases, may use original language titles
- Vinegar Syndrome: Genre-focused (horror, exploitation), uses catalog numbers
- Shout Factory/Scream Factory: Horror and cult films, collector's editions
- Kino Lorber: Classic cinema, international films
- Diabolik DVD: Underground and rare releases
- Grindhouse Releasing: Exploitation and grindhouse cinema
- Generic boutique: Blu-ray collector's editions

Create a concise search query that will work well with this retailer's search system.
Keep it simple - most retailer searches work best with 2-4 words.

Respond in this exact format:
QUERY: your optimized search query
REASONING: brief explanation of your choice"""

        try:
            response = self.client.chat.completions.create(
                model=MINI_MODEL,
                max_tokens=256,
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = response.choices[0].message.content
            result = self._parse_retailer_query_response(response_text)
            self._cache[cache_key] = result
            logger.debug(f"Retailer query for '{movie_title}' @ {retailer_name}: {result.query}")
            return result

        except Exception as e:
            logger.error(f"LLM retailer query tuning failed: {e}")
            raise

    def _parse_retailer_query_response(self, response: str) -> RetailerQueryResult:
        """Parse the LLM response into retailer query result."""
        lines = response.strip().split("\n")
        query = ""
        reasoning = ""

        for line in lines:
            line = line.strip()
            if line.startswith("QUERY:"):
                query = line.split(":", 1)[1].strip()
            elif line.startswith("REASONING:"):
                reasoning = line.split(":", 1)[1].strip()

        return RetailerQueryResult(query=query, reasoning=reasoning)

    def clear_cache(self):
        """Clear the in-memory cache."""
        self._cache.clear()
        logger.info("LLM response cache cleared")


# Alias for backwards compatibility
LLMService = OpenAIService
