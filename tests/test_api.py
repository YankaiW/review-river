"""End-to-end tests for the HTTP API in ``review_river.api``.

Requests go through real routing, JSON parsing, validation and
serialisation; the classifier is replaced by ``FakeClassifier`` via
``app.dependency_overrides``. Keyword logic is covered in
``test_classifier.py``, so these tests stay few and check the HTTP contract.
"""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from review_river.api import app
from review_river.deps import get_classifier
from review_river.domain import Category, Classification, Review


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
