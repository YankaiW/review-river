"""Unit tests for the classifier port, ``KeywordClassifier`` and ``LLMClassifier``."""

import logging

import httpx2
import pytest
from anthropic import APIConnectionError
from pydantic import BaseModel

from review_river.classifier import (
    SYSTEM_PROMPT,
    Classifier,
    ClassifierUnavailable,
    KeywordClassifier,
    LLMClassifier,
    LLMLabel,
    StructuredLLM,
)
from review_river.domain import Category, Classification, Review
from review_river.llm import LLMClient, LLMOutputError

VALID_LABEL: dict[str, object] = {
    "reason": "mentions a crash",
    "category": "bug",
    "confidence": 0.9,
}


class FakeLLM:
    """Stand-in for ``LLMClient``: records calls, returns a canned answer or raises.

    Parameters
    ----------
    answer : dict[str, object] or Exception
        Validated against the requested schema and returned, or raised as is.

    Attributes
    ----------
    calls : list[dict[str, object]]
        Keyword arguments of each ``parse`` call, in order.
    """

    def __init__(self, answer: dict[str, object] | Exception) -> None:
        self.answer = answer
        self.calls: list[dict[str, object]] = []

    async def parse[T: BaseModel](
        self, *, system: str, prompt: str, schema: type[T], max_tokens: int = 1024
    ) -> T:
        """Record the call, then return ``answer`` as ``schema`` or raise it.

        Parameters
        ----------
        system : str
            System prompt.
        prompt : str
            User turn.
        schema : type[T]
            Pydantic model to validate ``answer`` against.
        max_tokens : int, default 1024
            Output cap.

        Returns
        -------
        T
            ``answer`` validated as ``schema``.
        """
        self.calls.append(
            {"system": system, "prompt": prompt, "schema": schema, "max_tokens": max_tokens}
        )
        if isinstance(self.answer, Exception):
            raise self.answer
        return schema.model_validate(self.answer)


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


class TestLLMClassifier:
    """Request, mapping, error handling and logging of the LLM adapter."""

    async def test_maps_label_to_classification(self) -> None:
        """The model's category and confidence come back as a ``Classification``."""
        result = await LLMClassifier(FakeLLM(VALID_LABEL)).classify(Review(text="Crashes"))

        assert result == Classification(category=Category.BUG, confidence=0.9)

    async def test_sends_prompt_review_schema_and_max_tokens(self) -> None:
        """The system prompt, the tagged review, ``LLMLabel`` and ``max_tokens`` reach the LLM."""
        llm = FakeLLM(VALID_LABEL)

        await LLMClassifier(llm, max_tokens=16).classify(Review(text="Crashes on start"))

        assert llm.calls == [
            {
                "system": SYSTEM_PROMPT,
                "prompt": "<review>\nCrashes on start\n</review>",
                "schema": LLMLabel,
                "max_tokens": 16,
            }
        ]

    @pytest.mark.parametrize(
        "error",
        [
            LLMOutputError("stop_reason=refusal"),
            APIConnectionError(
                request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
            ),
        ],
        ids=["unusable-output", "provider-error"],
    )
    async def test_llm_failures_raise_classifier_unavailable(self, error: Exception) -> None:
        """Output and provider errors map to ``ClassifierUnavailable``, chained to the cause.

        Parameters
        ----------
        error : Exception
            Error the fake LLM raises.
        """
        with pytest.raises(ClassifierUnavailable) as exc_info:
            await LLMClassifier(FakeLLM(error)).classify(Review(text="Crashes"))

        assert exc_info.value.__cause__ is error

    async def test_unexpected_error_is_not_masked(self) -> None:
        """A bug in our code propagates as is instead of turning into a 503."""
        with pytest.raises(RuntimeError, match="bug in our code"):
            await LLMClassifier(FakeLLM(RuntimeError("bug in our code"))).classify(
                Review(text="Crashes")
            )

    async def test_out_of_range_confidence_is_rejected(self) -> None:
        """``confidence`` above 1 fails ``LLMLabel`` validation instead of passing through."""
        with pytest.raises(ValueError, match="less than or equal to 1"):
            await LLMClassifier(FakeLLM({**VALID_LABEL, "confidence": 1.2})).classify(
                Review(text="Crashes")
            )

    async def test_logs_label_and_reason(self, caplog: pytest.LogCaptureFixture) -> None:
        """One info line with category, confidence and reason; the review text is not logged.

        Parameters
        ----------
        caplog : pytest.LogCaptureFixture
            Captures log records emitted during the test.
        """
        caplog.set_level(logging.INFO, logger="review_river.classifier")

        await LLMClassifier(FakeLLM(VALID_LABEL)).classify(Review(text="Secret review text"))

        assert "category=bug" in caplog.text
        assert "confidence=0.90" in caplog.text
        assert "reason='mentions a crash'" in caplog.text
        assert "Secret review text" not in caplog.text


class TestClassifierProtocol:
    """Conformance of adapters to the ``Classifier`` port."""

    def test_keyword_classifier_satisfies_protocol(self) -> None:
        """``KeywordClassifier`` fits the ``Classifier`` port.

        The check is done by pyright on the annotated assignment; the assert only
        keeps the variable used.
        """
        classifier: Classifier = KeywordClassifier()

        assert classifier is not None

    def test_llm_classifier_satisfies_protocol(self) -> None:
        """``LLMClassifier`` fits the ``Classifier`` port (checked by pyright)."""
        classifier: Classifier = LLMClassifier(FakeLLM(VALID_LABEL))

        assert classifier is not None

    async def test_llm_client_satisfies_structured_llm(self) -> None:
        """``LLMClient`` fits the ``StructuredLLM`` port, checked by pyright."""
        llm: StructuredLLM = LLMClient(
            api_key="sk-ant-test",
            model="claude-opus-5-5",
            effort="low",
            timeout_s=5.0,
            max_retries=0,
        )

        assert llm is not None
        assert isinstance(llm, LLMClient)
        await llm.close()
