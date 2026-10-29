"""End-to-end tests for the HTTP API in ``review_river.api``.

Requests go through real routing, JSON parsing, validation and
serialisation; the classifier is replaced by ``FakeClassifier`` via
``app.dependency_overrides``. Keyword logic is covered in
``test_classifier.py``, so these tests stay few and check the HTTP contract.
"""

from collections.abc import Callable, Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel

from review_river.api import app
from review_river.classifier import ClassifierUnavailable
from review_river.deps import get_classifier, get_llm, get_settings
from review_river.domain import Category, Classification, Review
from review_river.llm import LLMClient, LLMOutputError


class FakeClassifier:
    """Test adapter: records what it receives and returns a fixed answer.

    The answer (``CONTENT``, 0.99) is one ``KeywordClassifier`` never gives,
    so seeing it in a response proves the override reached the endpoint.

    Attributes
    ----------
    seen : list[Review]
        Every review passed to ``classify``, in call order.
    """

    def __init__(self) -> None:
        self.seen: list[Review] = []

    async def classify(self, review: Review) -> Classification:
        """Record the review and return the fixed classification.

        Parameters
        ----------
        review : Review
            The review the endpoint passed in.

        Returns
        -------
        Classification
            Always ``CONTENT`` with confidence 0.99.
        """
        self.seen.append(review)
        return Classification(category=Category.CONTENT, confidence=0.99)


@pytest.fixture
def fake() -> FakeClassifier:
    """Provide a fresh ``FakeClassifier`` for one test.

    Returns
    -------
    FakeClassifier
        Shared with the ``client`` fixture within the same test.
    """
    return FakeClassifier()


@pytest.fixture
def client(fake: FakeClassifier) -> Iterator[TestClient]:
    """Provide a test client whose classifier dependency is the fake.

    Parameters
    ----------
    fake : FakeClassifier
        The fake to inject in place of ``get_classifier``.

    Yields
    ------
    TestClient
        Client for the app; overrides are cleared after the test.
    """
    app.dependency_overrides[get_classifier] = lambda: fake
    yield TestClient(app)
    app.dependency_overrides.clear()


class TestHealthEndpoint:
    """``GET /health``."""

    def test_health_returns_ok(self, client: TestClient) -> None:
        """``GET /health`` answers 200 with ``{"status": "ok"}``.

        Parameters
        ----------
        client : TestClient
            Client with the fake classifier injected.
        """
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestClassifyEndpoint:
    """``POST /classify``: injection, validation and production wiring."""

    def test_classify_returns_injected_classifier_answer(self, client: TestClient) -> None:
        """``POST /classify`` returns the fake's answer, not the keyword classifier's.

        Parameters
        ----------
        client : TestClient
            Client with the fake classifier injected.
        """
        response = client.post("/classify", json={"text": "Game crashes"})

        assert response.status_code == 200
        assert response.json() == {"category": "content", "confidence": 0.99}

    def test_classify_passes_review_to_classifier(
        self, client: TestClient, fake: FakeClassifier
    ) -> None:
        """The endpoint hands the parsed, stripped review to the classifier.

        Parameters
        ----------
        client : TestClient
            Client with the fake classifier injected.
        fake : FakeClassifier
            The same fake instance the client uses, inspected afterwards.
        """
        client.post("/classify", json={"text": "  Game crashes  "})

        assert fake.seen == [Review(text="Game crashes")]

    @pytest.mark.parametrize("body", [{"text": "   "}, {}, {"text": 123}])
    def test_invalid_body_is_rejected_before_classifier(
        self, client: TestClient, fake: FakeClassifier, body: dict[str, object]
    ) -> None:
        """Blank text, a missing field or a wrong type gives 422 and never reaches the classifier.

        Parameters
        ----------
        client : TestClient
            Client with the fake classifier injected.
        fake : FakeClassifier
            Checked to confirm it was never called.
        body : dict[str, object]
            Invalid JSON request body.
        """
        response = client.post("/classify", json=body)

        assert response.status_code == 422
        assert fake.seen == []

    def test_real_wiring_uses_keyword_classifier(self) -> None:
        """Without overrides, the production wiring classifies with ``KeywordClassifier``."""
        response = TestClient(app).post("/classify", json={"text": "Game crashes on startup"})

        assert response.status_code == 200
        assert response.json()["category"] == "bug"


@pytest.fixture
def llm_backend(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Switch the settings to the ``"llm"`` backend with a fake key for one test.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Sets the environment variables for this test only.

    Yields
    ------
    None
        Control passes to the test while ``get_settings`` reads the LLM settings.
    """
    monkeypatch.setenv("REVIEW_RIVER_CLASSIFIER_BACKEND", "llm")
    monkeypatch.setenv("REVIEW_RIVER_ANTHROPIC_API_KEY", "sk-ant-test")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


class TestLifespan:
    """Startup and shutdown of the shared LLM client.

    Only ``with TestClient(app)`` runs the lifespan; a bare ``TestClient(app)``
    skips it.
    """

    def test_keyword_backend_stores_no_client(self) -> None:
        """With the keyword backend, ``app.state.llm`` is ``None``."""
        with TestClient(app):
            assert app.state.llm is None

    @pytest.mark.usefixtures("llm_backend")
    def test_llm_backend_shares_one_client_and_closes_it(self) -> None:
        """One ``LLMClient`` is built at startup, stored on the app and closed on shutdown."""
        with TestClient(app):
            llm = app.state.llm
            assert isinstance(llm, LLMClient)
            assert not llm._client.is_closed()  # pyright: ignore[reportPrivateUsage]

        assert llm._client.is_closed()  # pyright: ignore[reportPrivateUsage]


class RaisingClassifier:
    """Test adapter whose ``classify`` always raises the given error.

    Parameters
    ----------
    error : Exception
        Raised on every call.
    """

    def __init__(self, error: Exception) -> None:
        self.error = error

    async def classify(self, review: Review) -> Classification:
        """Raise the configured error.

        Parameters
        ----------
        review : Review
            Ignored.

        Returns
        -------
        Classification
            Never returns.
        """
        raise self.error


class FailingLLM:
    """Stand-in for ``LLMClient`` whose ``parse`` always fails with no usable output."""

    async def parse[T: BaseModel](
        self, *, system: str, prompt: str, schema: type[T], max_tokens: int = 1024
    ) -> T:
        """Raise ``LLMOutputError`` as a ``max_tokens`` cut-off would.

        Parameters
        ----------
        system : str
            Ignored.
        prompt : str
            Ignored.
        schema : type[T]
            Ignored.
        max_tokens : int, default 1024
            Ignored.

        Returns
        -------
        T
            Never returns.
        """
        raise LLMOutputError("stop_reason=max_tokens")


@pytest.fixture
def overrides() -> Iterator[dict[Callable[..., Any], Callable[..., Any]]]:
    """Give a test the app's dependency overrides and clear them afterwards.

    Yields
    ------
    dict[Callable[..., Any], Callable[..., Any]]
        ``app.dependency_overrides``, to fill in the test.
    """
    yield app.dependency_overrides
    app.dependency_overrides.clear()


class TestClassifierErrors:
    """HTTP status of classifier failures: 503 for outages, 500 for bugs."""

    def test_unavailable_gives_503_without_internal_detail(
        self,
        overrides: dict[Callable[..., Any], Callable[..., Any]],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """``ClassifierUnavailable`` gives 503 with a generic body; the cause is only logged.

        Parameters
        ----------
        overrides : dict[Callable[..., Any], Callable[..., Any]]
            The app's dependency overrides.
        caplog : pytest.LogCaptureFixture
            Captures log records emitted during the test.
        """
        error = ClassifierUnavailable("LLM provider call failed")
        error.__cause__ = RuntimeError("internal provider detail")
        overrides[get_classifier] = lambda: RaisingClassifier(error)

        response = TestClient(app).post("/classify", json={"text": "Game crashes"})

        assert response.status_code == 503
        assert response.json() == {"detail": "classifier unavailable, try again later"}
        assert response.headers["retry-after"] == "5"
        assert "internal provider detail" not in response.text
        assert "classifier_unavailable path=/classify" in caplog.text
        assert "internal provider detail" in caplog.text

    def test_unexpected_error_gives_500(
        self, overrides: dict[Callable[..., Any], Callable[..., Any]]
    ) -> None:
        """Any other exception is a bug and stays a 500, not a 503.

        Parameters
        ----------
        overrides : dict[Callable[..., Any], Callable[..., Any]]
            The app's dependency overrides.
        """
        overrides[get_classifier] = lambda: RaisingClassifier(RuntimeError("bug"))

        client = TestClient(app, raise_server_exceptions=False)
        response = client.post("/classify", json={"text": "Game crashes"})

        assert response.status_code == 500

    @pytest.mark.usefixtures("llm_backend")
    def test_llm_output_error_reaches_client_as_503(
        self, overrides: dict[Callable[..., Any], Callable[..., Any]]
    ) -> None:
        """Full chain: ``LLMOutputError`` → ``LLMClassifier`` → ``ClassifierUnavailable`` → 503.

        Only the LLM is faked; settings, ``get_classifier``, the adapter and
        the handler are real.

        Parameters
        ----------
        overrides : dict[Callable[..., Any], Callable[..., Any]]
            The app's dependency overrides.
        """
        overrides[get_llm] = FailingLLM

        response = TestClient(app).post("/classify", json={"text": "Game crashes"})

        assert response.status_code == 503
