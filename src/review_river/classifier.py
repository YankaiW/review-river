"""Classifier port and its rule-based keyword adapter."""

import re
from collections.abc import Mapping
from typing import Protocol

from review_river.domain import Category, Classification, Review

DEFAULT_KEYWORDS: Mapping[Category, tuple[str, ...]] = {
    Category.BUG: ("crash", "bug", "freeze", "error"),
    Category.PERFORMANCE: ("fps", "lag", "stutter"),
    Category.PAYMENTS: ("refund", "price", "microtransaction", "pay to win"),
    Category.CONTENT: ("story", "levels", "dlc"),
}


class Classifier(Protocol):
    """Port: anything that can turn a ``Review`` into a ``Classification``."""

    async def classify(self, review: Review) -> Classification:
        """Classify one review.

        Parameters
        ----------
        review : Review
            The review to classify.

        Returns
        -------
        Classification
            Assigned category and confidence.
        """
        ...


class KeywordClassifier:
    """Rule-based adapter: the first category whose keyword appears in the text wins.

    A keyword must start a word but may be followed by more letters, so
    ``"crash"`` matches ``"crashes"`` and ``"lag"`` matches ``"lagging"``, but
    ``"lag"`` does not match ``"flagship"``. Categories are checked in mapping
    order, which sets their priority.

    Parameters
    ----------
    keywords : Mapping[Category, tuple[str, ...]] or None, optional
        Keywords per category, checked in insertion order. Defaults to
        ``DEFAULT_KEYWORDS``.

    Attributes
    ----------
    MATCH_CONFIDENCE : float
        Confidence returned when a keyword matches. Fixed, because keyword
        rules carry no real probability.
    FALLBACK_CONFIDENCE : float
        Confidence returned with ``Category.OTHER`` when nothing matches.
    """

    MATCH_CONFIDENCE = 0.6
    FALLBACK_CONFIDENCE = 0.2

    def __init__(self, keywords: Mapping[Category, tuple[str, ...]] | None = None) -> None:
        self._keywords = keywords if keywords is not None else DEFAULT_KEYWORDS

    async def classify(self, review: Review) -> Classification:
        """Classify one review by keyword matching.

        Parameters
        ----------
        review : Review
            The review to classify.

        Returns
        -------
        Classification
            The first matching category with ``MATCH_CONFIDENCE``, or
            ``Category.OTHER`` with ``FALLBACK_CONFIDENCE`` if nothing matches.
        """
        text = review.text.lower()
        for category, words in self._keywords.items():
            for word in words:
                if re.search(rf"\b{re.escape(word)}", text):
                    return Classification(category=category, confidence=self.MATCH_CONFIDENCE)
        return Classification(category=Category.OTHER, confidence=self.FALLBACK_CONFIDENCE)
