"""Unit tests for the domain model in ``review_river.domain``."""

import pytest
from pydantic import ValidationError

from review_river.domain import Category, Classification, Review


class TestReview:
    """Validation rules of ``Review``."""

    def test_valid_text_is_accepted(self) -> None:
        """Non-empty text is accepted and surrounding whitespace is stripped."""
        review = Review(text="  Game crashes on startup ")

        assert review.text == "Game crashes on startup"

    @pytest.mark.parametrize("text", ["", "   ", "\n\t"])
    def test_empty_or_blank_text_is_rejected(self, text: str) -> None:
        """Empty text, or text that is empty after stripping, is rejected.

        Parameters
        ----------
        text : str
            Review text that must fail validation.
        """
        with pytest.raises(ValidationError):
            Review(text=text)


class TestClassification:
    """Validation rules and immutability of ``Classification``."""

    def test_string_category_is_converted_to_enum(self) -> None:
        """Raw input such as LLM JSON gets its category string turned into a member."""
        result = Classification.model_validate({"category": "bug", "confidence": 0.6})

        assert result.category is Category.BUG

    def test_unknown_category_is_rejected(self) -> None:
        """A category outside ``Category`` is rejected."""
        with pytest.raises(ValidationError):
            Classification.model_validate({"category": "lag", "confidence": 0.6})

    @pytest.mark.parametrize("confidence", [0.0, 0.5, 1.0])
    def test_confidence_inside_bounds_is_accepted(self, confidence: float) -> None:
        """Confidence on or between the bounds 0 and 1 is accepted.

        Parameters
        ----------
        confidence : float
            Value inside ``[0, 1]``.
        """
        result = Classification(category=Category.BUG, confidence=confidence)

        assert result.confidence == confidence

    @pytest.mark.parametrize("confidence", [-0.1, 1.5])
    def test_confidence_outside_bounds_is_rejected(self, confidence: float) -> None:
        """Confidence below 0 or above 1 is rejected.

        Parameters
        ----------
        confidence : float
            Value outside ``[0, 1]``.
        """
        with pytest.raises(ValidationError):
            Classification(category=Category.BUG, confidence=confidence)

    def test_is_immutable(self) -> None:
        """Assigning to an attribute after creation is rejected (frozen value object)."""
        result = Classification(category=Category.BUG, confidence=0.6)

        with pytest.raises(ValidationError):
            result.confidence = 0.9
