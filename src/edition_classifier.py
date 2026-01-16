"""
Edition classifier using OpenAI for intelligent product classification.
Identifies special editions, formats (4K, Blu-ray, DVD), and boutique labels.
"""

import os
import json
import logging
from typing import Tuple, Optional
from dataclasses import dataclass

from openai import OpenAI

logger = logging.getLogger(__name__)

# Known boutique labels for reference
BOUTIQUE_LABELS = [
    "Criterion Collection", "Arrow Video", "Arrow Academy",
    "Kino Lorber", "Kino Classics", "BFI", "Masters of Cinema",
    "Eureka", "Shout Factory", "Scream Factory", "Vinegar Syndrome",
    "Indicator", "Second Sight", "Blue Underground", "88 Films",
    "Severin Films", "Imprint", "Studio Canal", "Warner Archive",
    "Twilight Time", "Olive Films", "Cohen Film Collection",
    "Film Movement", "Flicker Alley", "Milestone Films",
    "Oscilloscope", "Music Box Films", "Drafthouse Films",
    "AGFA", "Synapse Films", "Code Red", "Unearthed Films",
    "MVD Rewind", "Fun City Editions", "Arbelos", "Deaf Crocodile"
]


@dataclass
class ClassificationResult:
    """Result of edition classification."""
    is_special_edition: bool
    confidence: float
    format: str  # "4K UHD", "Blu-ray", "DVD", "Unknown"
    label: Optional[str]  # Boutique label if identified
    edition_type: Optional[str]  # "Collector's", "Limited", "Criterion", etc.
    reason: str  # Explanation for the classification


class EditionClassifier:
    """
    Uses OpenAI to classify movie product listings.
    Determines if a product is a special/collector's edition worth tracking.
    """

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self._system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        """Build the system prompt for classification."""
        labels_str = ", ".join(BOUTIQUE_LABELS[:20]) + ", etc."

        return f"""You are a movie physical media expert. Your job is to analyze product listings and determine if they are special/collector's editions worth tracking for deals.

SPECIAL EDITIONS include:
- Boutique label releases (e.g., {labels_str})
- Collector's editions, Limited editions, Steelbooks
- Restored/Remastered releases with special features
- Anniversary editions, Director's cuts with extras
- Box sets and complete collections

NOT SPECIAL EDITIONS:
- Standard retail releases (regular Blu-ray/DVD without special features)
- Digital codes or streaming
- Used/pre-owned standard editions
- Bootlegs or unauthorized releases
- VHS tapes
- Standard slipcovers without additional content

FORMATS to identify:
- "4K UHD" - 4K Ultra HD Blu-ray
- "Blu-ray" - Standard Blu-ray
- "DVD" - DVD format
- "Combo" - Multiple formats included
- "Unknown" - Cannot determine

Respond ONLY with valid JSON in this exact format:
{{
  "is_special_edition": true/false,
  "confidence": 0.0-1.0,
  "format": "4K UHD" | "Blu-ray" | "DVD" | "Combo" | "Unknown",
  "label": "Label name or null",
  "edition_type": "Type description or null",
  "reason": "Brief explanation"
}}"""

    def classify(self, product_title: str) -> ClassificationResult:
        """
        Classify a product title using OpenAI.

        Returns ClassificationResult with details about the edition.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._system_prompt},
                    {"role": "user", "content": f"Classify this product: {product_title}"}
                ],
                temperature=0.1,  # Low temperature for consistent classification
                max_tokens=200,
            )

            result_text = response.choices[0].message.content.strip()

            # Parse JSON response
            # Handle potential markdown code blocks
            if result_text.startswith("```"):
                result_text = result_text.split("```")[1]
                if result_text.startswith("json"):
                    result_text = result_text[4:]
                result_text = result_text.strip()

            data = json.loads(result_text)

            return ClassificationResult(
                is_special_edition=data.get("is_special_edition", False),
                confidence=float(data.get("confidence", 0.0)),
                format=data.get("format", "Unknown"),
                label=data.get("label"),
                edition_type=data.get("edition_type"),
                reason=data.get("reason", ""),
            )

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse OpenAI response: {e}")
            logger.debug(f"Response was: {result_text}")
            return ClassificationResult(
                is_special_edition=False,
                confidence=0.0,
                format="Unknown",
                label=None,
                edition_type=None,
                reason="Failed to parse classification response",
            )
        except Exception as e:
            logger.error(f"OpenAI classification failed: {e}")
            return ClassificationResult(
                is_special_edition=False,
                confidence=0.0,
                format="Unknown",
                label=None,
                edition_type=None,
                reason=f"Classification error: {str(e)}",
            )

    def is_special_edition(self, product_title: str) -> Tuple[bool, float, str]:
        """
        Compatibility method matching EditionMatcher interface.

        Returns (is_match, confidence_score, description).
        """
        result = self.classify(product_title)

        description = result.reason
        if result.label:
            description = f"{result.label} - {result.edition_type or 'Special Edition'}"
        elif result.edition_type:
            description = result.edition_type

        return (result.is_special_edition, result.confidence, description)


def create_classifier(api_key: Optional[str] = None) -> EditionClassifier:
    """Factory function to create a classifier."""
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY", "")

    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")

    return EditionClassifier(api_key=api_key)


if __name__ == "__main__":
    # Test the classifier
    logging.basicConfig(level=logging.INFO)

    test_products = [
        "The Shining (Criterion Collection) [4K UHD Blu-ray]",
        "Jaws - Standard Blu-ray",
        "Alien 4K Ultra HD Steelbook Limited Edition",
        "Spider-Man DVD Walmart Exclusive",
        "Seven Samurai (Criterion Collection) Blu-ray",
        "Arrow Video: Society Limited Edition Blu-ray with Slipcover",
        "The Matrix - Regular DVD",
    ]

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Set OPENAI_API_KEY to test")
    else:
        classifier = EditionClassifier(api_key=api_key)

        print("\nClassification Results:\n")
        for product in test_products:
            result = classifier.classify(product)
            status = "SPECIAL" if result.is_special_edition else "standard"
            print(f"[{status:8}] {result.confidence:.0%} | {product}")
            print(f"           Format: {result.format}, Label: {result.label}")
            print(f"           Reason: {result.reason}\n")
