"""Unit tests for the dependency providers in ``review_river.deps``."""

from collections.abc import Iterator

import pytest

from review_river.classifier import KeywordClassifier
from review_river.config import Settings
from review_river.deps import get_classifier, get_settings


@pytest.fixture(autouse=True)
def clear_settings_cache() -> Iterator[None]:
    """Empty the ``get_settings`` cache before and after every test.

    Yields
    ------
    None
        Control passes to the test while the cache is empty.
    """
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_get_settings_returns_cached_instance() -> None:
    """Repeated calls return the very same ``Settings`` object."""
    assert get_settings() is get_settings()


def test_cache_clear_rereads_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """After ``cache_clear()`` a changed environment is picked up.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Sets the environment variable for this test only.
    """
    before = get_settings()
    monkeypatch.setenv("REVIEW_RIVER_APP_NAME", "renamed")
    get_settings.cache_clear()

    after = get_settings()

    assert after is not before
    assert after.app_name == "renamed"


def test_keyword_backend_gives_keyword_classifier() -> None:
    """The ``"keyword"`` backend is wired to ``KeywordClassifier``."""
    settings = Settings(classifier_backend="keyword")

    assert isinstance(get_classifier(settings), KeywordClassifier)
