"""
Edition matcher using sentence-transformers for semantic similarity.
Matches product listings against example special editions.
"""

import logging
from typing import List, Tuple, Optional
from pathlib import Path

import yaml
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EditionMatcher:
    """
    Uses sentence embeddings to determine if a product listing
    matches the style of collector's/special editions.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        examples_path: Optional[Path] = None,
        similarity_threshold: float = 0.5,
    ):
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.model: Optional[SentenceTransformer] = None
        self.example_embeddings: Optional[np.ndarray] = None
        self.examples: List[str] = []

        # Load examples if path provided
        if examples_path:
            self.load_examples(examples_path)

    def _ensure_model_loaded(self):
        """Lazy load the model to save memory when not needed."""
        if self.model is None:
            logger.info(f"Loading sentence transformer model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)

    def load_examples(self, examples_path: Path):
        """Load example edition descriptions from YAML file."""
        try:
            with open(examples_path, "r") as f:
                data = yaml.safe_load(f)
                self.examples = data.get("examples", [])

            if self.examples:
                self._ensure_model_loaded()
                logger.info(f"Computing embeddings for {len(self.examples)} examples")
                self.example_embeddings = self.model.encode(
                    self.examples, convert_to_numpy=True
                )
            else:
                logger.warning("No examples found in examples file")

        except Exception as e:
            logger.error(f"Failed to load examples: {e}")
            raise

    def add_examples(self, examples: List[str]):
        """Add examples programmatically."""
        self.examples.extend(examples)
        self._ensure_model_loaded()
        self.example_embeddings = self.model.encode(
            self.examples, convert_to_numpy=True
        )

    def compute_similarity(self, text: str) -> Tuple[float, str]:
        """
        Compute max similarity between text and example editions.
        Returns (max_similarity_score, closest_example).
        """
        if not self.examples or self.example_embeddings is None:
            raise ValueError("No examples loaded. Call load_examples() first.")

        self._ensure_model_loaded()

        # Encode the input text
        text_embedding = self.model.encode([text], convert_to_numpy=True)

        # Compute cosine similarities with all examples
        similarities = np.dot(self.example_embeddings, text_embedding.T).flatten()

        # Normalize (embeddings from sentence-transformers are already normalized)
        max_idx = np.argmax(similarities)
        max_similarity = float(similarities[max_idx])
        closest_example = self.examples[max_idx]

        return max_similarity, closest_example

    def is_special_edition(self, product_title: str) -> Tuple[bool, float, str]:
        """
        Determine if a product title matches a special edition pattern.
        Returns (is_match, similarity_score, closest_example).
        """
        similarity, closest = self.compute_similarity(product_title)
        is_match = similarity >= self.similarity_threshold

        if is_match:
            logger.debug(
                f"Match found: '{product_title}' "
                f"(similarity: {similarity:.3f}, closest: '{closest}')"
            )
        else:
            logger.debug(
                f"No match: '{product_title}' (similarity: {similarity:.3f})"
            )

        return is_match, similarity, closest

    def rank_products(
        self, products: List[str]
    ) -> List[Tuple[str, float, str, bool]]:
        """
        Rank a list of products by their similarity to special editions.
        Returns list of (product, similarity, closest_example, is_match).
        """
        results = []
        for product in products:
            similarity, closest = self.compute_similarity(product)
            is_match = similarity >= self.similarity_threshold
            results.append((product, similarity, closest, is_match))

        # Sort by similarity descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results


def create_matcher(
    config_dir: Path, similarity_threshold: float = 0.5
) -> EditionMatcher:
    """Factory function to create a matcher with config."""
    examples_path = config_dir / "examples.yaml"
    return EditionMatcher(
        examples_path=examples_path, similarity_threshold=similarity_threshold
    )


if __name__ == "__main__":
    # Test the matcher
    logging.basicConfig(level=logging.INFO)

    # Create matcher with test examples
    matcher = EditionMatcher(similarity_threshold=0.5)
    matcher.add_examples(
        [
            "Criterion Collection Blu-ray with new 4K digital restoration",
            "Arrow Video Limited Edition Blu-ray with booklet",
            "BFI 4K Ultra HD Blu-ray remastered edition",
        ]
    )

    # Test products
    test_products = [
        "House (Criterion Collection) [Blu-ray]",
        "House DVD Standard Edition",
        "The Holy Mountain Arrow Video Blu-ray Limited Edition",
        "Random Movie Walmart Exclusive DVD",
        "Vertigo 4K UHD Criterion Collection",
        "Spider-Man Blu-ray",
    ]

    print("\nProduct Matching Results:\n")
    results = matcher.rank_products(test_products)
    for product, score, closest, is_match in results:
        status = "MATCH" if is_match else "no match"
        print(f"[{status:8}] {score:.3f} | {product}")
        print(f"           Closest: {closest}\n")
