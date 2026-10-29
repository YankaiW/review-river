"""Unit tests for the dependency providers in ``review_river.deps``."""

from collections.abc import Iterator

import pytest
from pydantic import SecretStr

from review_river.classifier import KeywordClassifier, LLMClassifier
from review_river.config import Settings
from review_river.deps import build_llm, get_classifier, get_settings
from review_river.llm import LLMClient


@pytest.fixture
def llm_settings() -> Settings:
    """Provide settings for the ``"llm"`` backend with a fake key.

    Built inside the test, after ``conftest`` has isolated the environment,
    so the developer's ``.env`` cannot leak in.

    Returns
    -------
    Settings
        LLM backend, fake key, no retries.
    """
    return Settings(
        classifier_backend="llm", anthropic_api_key=SecretStr("sk-ant-test"), llm_max_retries=0
    )


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


class TestGetSettings:
    """Caching behaviour of ``get_settings``."""

    def test_get_settings_returns_cached_instance(self) -> None:
        """Repeated calls return the very same ``Settings`` object."""
        assert get_settings() is get_settings()

    def test_cache_clear_rereads_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
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


class TestGetClassifier:
    """Backend-to-adapter wiring of ``get_classifier``."""

    def test_keyword_backend_gives_keyword_classifier(self) -> None:
        """The ``"keyword"`` backend is wired to ``KeywordClassifier``."""
        settings = Settings(classifier_backend="keyword")

        assert isinstance(get_classifier(settings, llm=None), KeywordClassifier)

    async def test_llm_backend_wraps_the_shared_client(self, llm_settings: Settings) -> None:
        """The ``"llm"`` backend wraps the client it is given in ``LLMClassifier``.

        Parameters
        ----------
        llm_settings : Settings
            Settings for the LLM backend.
        """
        llm = build_llm(llm_settings)
        assert llm is not None

        classifier = get_classifier(llm_settings, llm=llm)

        assert isinstance(classifier, LLMClassifier)
        await llm.close()

    def test_llm_backend_without_client_raises(self, llm_settings: Settings) -> None:
        """A missing client is a wiring bug: ``RuntimeError``, not a 503.

        Parameters
        ----------
        llm_settings : Settings
            Settings for the LLM backend.
        """
        with pytest.raises(RuntimeError, match="lifespan"):
            get_classifier(llm_settings, llm=None)


class TestBuildLLM:
    """Client construction from settings in ``build_llm``."""

    def test_keyword_backend_builds_no_client(self) -> None:
        """The keyword backend needs no client and no key."""
        assert build_llm(Settings(classifier_backend="keyword")) is None

    async def test_llm_backend_builds_client(self, llm_settings: Settings) -> None:
        """The ``"llm"`` backend gets an ``LLMClient``; building it makes no request.

        Parameters
        ----------
        llm_settings : Settings
            Settings for the LLM backend.
        """
        llm = build_llm(llm_settings)

        assert isinstance(llm, LLMClient)
        await llm.close()
