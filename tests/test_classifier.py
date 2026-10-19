"""Unit tests for the classifier port and ``KeywordClassifier``."""

import pytest

from review_river.classifier import Classifier, KeywordClassifier
from review_river.domain import Category, Review


class TestKeywordClassifier:
    """Behaviour of the rule-based keyword adapter."""

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("Game crashes on startup", Category.BUG),
            ("Very buggy since the patch", Category.BUG),
            ("Constant lagging and low FPS", Category.PERFORMANCE),
            ("Way too pay to win", Category.PAYMENTS),
            ("I want a refund", Category.PAYMENTS),
            ("The story is amazing", Category.CONTENT),
            ("Lovely art style", Category.OTHER),
        ],
    )
    async def test_classifies_by_keyword(self, text: str, expected: Category) -> None:
        """Each review lands in the category of the keyword it contains.

        Parameters
        ----------
        text : str
            Review text.
        expected : Category
            Category the review should be assigned.
        """
        result = await KeywordClassifier().classify(Review(text=text))

        assert result.category is expected

    async def test_matching_is_case_insensitive(self) -> None:
        """Upper-case keywords match the same as lower-case ones."""
        result = await KeywordClassifier().classify(Review(text="CRASH TO DESKTOP"))

        assert result.category is Category.BUG

    async def test_keyword_inside_a_word_does_not_match(self) -> None:
        """A keyword in the middle of a word is ignored: "lag" is not in "flagship"."""
        result = await KeywordClassifier().classify(Review(text="The flagship mode is great"))

        assert result.category is Category.OTHER

    async def test_earlier_category_wins_on_multiple_matches(self) -> None:
        """When several categories match, the first one in the mapping wins."""
        result = await KeywordClassifier().classify(Review(text="Crashes after I bought the DLC"))

        assert result.category is Category.BUG

    async def test_match_uses_match_confidence(self) -> None:
        """A keyword match returns ``MATCH_CONFIDENCE``."""
        result = await KeywordClassifier().classify(Review(text="Game crashes"))

        assert result.confidence == KeywordClassifier.MATCH_CONFIDENCE

    async def test_no_match_falls_back_to_other(self) -> None:
        """No keyword match returns ``Category.OTHER`` with ``FALLBACK_CONFIDENCE``."""
        result = await KeywordClassifier().classify(Review(text="Lovely art style"))

        assert result.category is Category.OTHER
        assert result.confidence == KeywordClassifier.FALLBACK_CONFIDENCE

    async def test_custom_keywords_replace_defaults(self) -> None:
        """Keywords passed to the constructor replace ``DEFAULT_KEYWORDS``."""
        classifier = KeywordClassifier({Category.CONTENT: ("dragon",)})

        dragon = await classifier.classify(Review(text="Epic dragon fight"))
        crash = await classifier.classify(Review(text="Game crashes"))

        assert dragon.category is Category.CONTENT
        assert crash.category is Category.OTHER


def test_keyword_classifier_satisfies_protocol() -> None:
    """``KeywordClassifier`` fits the ``Classifier`` port.

    The check is done by pyright on the annotated assignment; the assert only
    keeps the variable used.
    """
    classifier: Classifier = KeywordClassifier()

    assert classifier is not None
