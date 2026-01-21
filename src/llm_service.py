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


@dataclass
class MovieSuggestion:
    """A movie suggestion from LLM interpretation."""
    title: str
    year: Optional[int]
    reason: str


@dataclass
class MovieSuggestionsResult:
    """Result from LLM movie suggestions."""
    suggestions: List[MovieSuggestion]
    interpreted_query: str


@dataclass
class BatchValidationResult:
    """Result from batch validation of search results."""
    valid_indices: List[int]
    invalid_indices: List[int]
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

    def suggest_movies(self, user_input: str) -> MovieSuggestionsResult:
        """
        Interpret user input and suggest matching movies.

        Handles partial input, misspellings, and ambiguous queries.
        Returns normalized movie titles that can be looked up in TMDB.

        Args:
            user_input: Partial or complete movie search input

        Returns:
            MovieSuggestionsResult with suggested movies
        """
        cache_key = self._get_cache_key("suggest", user_input.lower())
        if cache_key in self._cache:
            logger.debug(f"Cache hit for movie suggestion: {user_input}")
            return self._cache[cache_key]

        prompt = f"""A user is searching for a movie to find special edition Blu-ray/4K releases.
Interpret their input and suggest the most likely movies they're looking for.

User input: "{user_input}"

Consider:
- Partial titles (e.g., "seven sam" → "Seven Samurai")
- Misspellings (e.g., "shawshank redemtion" → "The Shawshank Redemption")
- Common nicknames (e.g., "alien 3" → "Alien³")
- Ambiguous titles - suggest multiple if unclear
- The user is likely looking for cult classics, horror, sci-fi, or films with boutique releases

Suggest up to 5 movies, prioritizing films that typically have collector's editions.

Respond in this exact format:
INTERPRETED: what you think they're searching for
MOVIE1: Title (Year) | brief reason
MOVIE2: Title (Year) | brief reason
MOVIE3: Title (Year) | brief reason
(continue for up to 5 movies)

Example:
INTERPRETED: The Thing by John Carpenter
MOVIE1: The Thing (1982) | John Carpenter's sci-fi horror classic, many special editions
MOVIE2: The Thing from Another World (1951) | Original film, Criterion release available"""

        try:
            response = self.client.chat.completions.create(
                model=MINI_MODEL,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = response.choices[0].message.content
            result = self._parse_movie_suggestions_response(response_text)
            self._cache[cache_key] = result
            logger.info(f"LLM suggested {len(result.suggestions)} movies for '{user_input}'")
            return result

        except Exception as e:
            logger.error(f"LLM movie suggestion failed: {e}")
            raise

    def _parse_movie_suggestions_response(self, response: str) -> MovieSuggestionsResult:
        """Parse the LLM response into movie suggestions."""
        import re

        lines = response.strip().split("\n")
        suggestions = []
        interpreted = ""

        for line in lines:
            line = line.strip()
            if line.startswith("INTERPRETED:"):
                interpreted = line.split(":", 1)[1].strip()
            elif line.startswith("MOVIE") and ":" in line:
                # Parse "Title (Year) | reason" format
                content = line.split(":", 1)[1].strip()
                if "|" in content:
                    title_part, reason = content.split("|", 1)
                    title_part = title_part.strip()
                    reason = reason.strip()
                else:
                    title_part = content
                    reason = ""

                # Extract year from "Title (Year)"
                year_match = re.search(r'\((\d{4})\)$', title_part)
                if year_match:
                    year = int(year_match.group(1))
                    title = title_part[:year_match.start()].strip()
                else:
                    year = None
                    title = title_part

                if title:
                    suggestions.append(MovieSuggestion(
                        title=title,
                        year=year,
                        reason=reason
                    ))

        return MovieSuggestionsResult(
            suggestions=suggestions,
            interpreted_query=interpreted
        )

    def batch_validate_results(
        self,
        movie_title: str,
        year: Optional[int],
        director: Optional[str],
        product_titles: List[str]
    ) -> BatchValidationResult:
        """
        Validate a batch of search results against the target movie.

        More efficient than individual validation for post-search filtering.

        Args:
            movie_title: Target movie title
            year: Target movie year
            director: Target movie director
            product_titles: List of product listing titles to validate

        Returns:
            BatchValidationResult with indices of valid/invalid results
        """
        if not product_titles:
            return BatchValidationResult(
                valid_indices=[],
                invalid_indices=[],
                reasoning="No products to validate"
            )

        cache_key = self._get_cache_key(
            "batch_validate",
            movie_title,
            year,
            hash(tuple(product_titles[:10]))  # Cache based on first 10 products
        )
        if cache_key in self._cache:
            logger.debug(f"Cache hit for batch validation: {movie_title}")
            return self._cache[cache_key]

        year_info = f" ({year})" if year else ""
        director_info = f"Director: {director}" if director else ""

        # Format product list with indices
        products_list = "\n".join(
            f"{i}: {title}" for i, title in enumerate(product_titles)
        )

        prompt = f"""Validate which of these product listings are for the EXACT movie specified.

TARGET MOVIE:
Title: {movie_title}{year_info}
{director_info}

PRODUCT LISTINGS:
{products_list}

STRICT VALIDATION RULES:
- The product MUST be for the exact movie title specified, not a sequel, prequel, reboot, or different film in the same franchise
- For franchises (Spider-Man, Batman, Star Wars, etc.), match ONLY the specific film - "Spider-Man (2002)" is NOT "Spider-Man 2", "The Amazing Spider-Man", or "Spider-Man: Homecoming"
- Year must match if specified - a 2002 film is NOT a 2012 reboot
- Multi-packs or collections only count if they include the target film
- When in doubt, mark as INVALID

Respond in this exact format:
VALID: comma-separated indices of matching products (e.g., 0, 2, 3, 5) or "none" if no matches
INVALID: comma-separated indices of non-matching products (e.g., 1, 4)
REASONING: brief explanation of rejections"""

        try:
            response = self.client.chat.completions.create(
                model=MINI_MODEL,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = response.choices[0].message.content
            result = self._parse_batch_validation_response(response_text, len(product_titles))
            self._cache[cache_key] = result
            logger.info(f"Batch validation: {len(result.valid_indices)} valid, {len(result.invalid_indices)} invalid")
            return result

        except Exception as e:
            logger.error(f"LLM batch validation failed: {e}")
            raise

    def _parse_batch_validation_response(
        self,
        response: str,
        total_count: int
    ) -> BatchValidationResult:
        """Parse the LLM response into batch validation result."""
        lines = response.strip().split("\n")
        valid_indices = []
        invalid_indices = []
        reasoning = ""

        for line in lines:
            line = line.strip()
            if line.startswith("VALID:"):
                indices_str = line.split(":", 1)[1].strip()
                if indices_str and indices_str.lower() != "none":
                    for idx in indices_str.split(","):
                        try:
                            valid_indices.append(int(idx.strip()))
                        except ValueError:
                            pass
            elif line.startswith("INVALID:"):
                indices_str = line.split(":", 1)[1].strip()
                if indices_str and indices_str.lower() != "none":
                    for idx in indices_str.split(","):
                        try:
                            invalid_indices.append(int(idx.strip()))
                        except ValueError:
                            pass
            elif line.startswith("REASONING:"):
                reasoning = line.split(":", 1)[1].strip()

        # Validate indices are in range
        valid_indices = [i for i in valid_indices if 0 <= i < total_count]
        invalid_indices = [i for i in invalid_indices if 0 <= i < total_count]

        return BatchValidationResult(
            valid_indices=valid_indices,
            invalid_indices=invalid_indices,
            reasoning=reasoning
        )

    def clear_cache(self):
        """Clear the in-memory cache."""
        self._cache.clear()
        logger.info("LLM response cache cleared")


# Alias for backwards compatibility
LLMService = OpenAIService
