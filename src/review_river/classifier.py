"""Classifier port and its rule-based keyword adapter."""

import re
from collections.abc import Mapping
from typing import Protocol

from pydantic import BaseModel, Field

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


class ClassifierUnavailable(Exception):
    """A classifier exists but cannot answer right now.

    Part of the port's contract: adapters raise it for backend failures
    (provider down, timeout, unusable output), so callers handle one error
    without knowing which backend is behind the port.
    """


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


class LLMLabel(BaseModel):
    """Output contract of the LLM adapter, not a domain type.

    The model fills the fields in order, so ``reason`` comes first and the
    label follows from it. The API enforces the ``Category`` enum; the
    ``confidence`` bounds are checked locally by pydantic.

    Attributes
    ----------
    reason : str
        One-sentence justification, logged for error analysis.
    category : Category
        Assigned category.
    confidence : float
        Self-reported confidence in [0, 1].
    """

    reason: str
    category: Category
    confidence: float = Field(ge=0.0, le=1.0)


class StructuredLLM(Protocol):
    """Port for an LLM that returns answers validated against a pydantic schema."""

    async def parse[T: BaseModel](
        self, *, system: str, prompt: str, schema: type[T], max_tokens: int = 1024
    ) -> T:
        """Ask the model for one answer that validates against ``schema``.

        Parameters
        ----------
        system : str
            System prompt: task, definitions, examples.
        prompt : str
            User turn, e.g. the review text.
        schema : type[T]
            Pydantic model the answer must validate against.
        max_tokens : int, default 1024
            Output cap; thinking tokens count against it.

        Returns
        -------
        T
            The validated instance of ``schema``.
        """
        ...
