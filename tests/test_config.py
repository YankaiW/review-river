"""Unit tests for ``Settings`` in ``review_river.config``.

These cover our own decisions (defaults, the env prefix, allowed backends),
not pydantic-settings itself. The autouse fixture in ``conftest.py`` keeps
the shell environment and any local ``.env`` out of the tests.
"""

import pytest
from pydantic import ValidationError

from review_river.config import Settings


class TestSettings:
    """Defaults, env prefix and validation of ``Settings``."""

    def test_defaults_without_environment(self) -> None:
        """With no environment variables, every field takes its default."""
        settings = Settings()

        assert settings.app_name == "review-river"
        assert settings.classifier_backend == "keyword"

    def test_prefixed_env_var_overrides_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``REVIEW_RIVER_APP_NAME`` overrides the default ``app_name``.

        Parameters
        ----------
        monkeypatch : pytest.MonkeyPatch
            Sets the environment variable for this test only.
        """
        monkeypatch.setenv("REVIEW_RIVER_APP_NAME", "test-app")

        assert Settings().app_name == "test-app"

    def test_unprefixed_env_var_is_ignored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A plain ``APP_NAME`` without the prefix does not leak into the settings.

        Parameters
        ----------
        monkeypatch : pytest.MonkeyPatch
            Sets the environment variable for this test only.
        """
        monkeypatch.setenv("APP_NAME", "someone-elses-app")

        assert Settings().app_name == "review-river"

    def test_unknown_backend_is_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An unknown ``classifier_backend`` fails at creation, so the app cannot start.

        Parameters
        ----------
        monkeypatch : pytest.MonkeyPatch
            Sets the environment variable for this test only.
        """
        monkeypatch.setenv("REVIEW_RIVER_CLASSIFIER_BACKEND", "bogus")

        with pytest.raises(ValidationError):
            Settings()

    def test_llm_defaults_without_environment(self) -> None:
        """With no environment variables, the LLM fields take their defaults and no key is set."""
        settings = Settings()

        assert settings.llm_model == "claude-opus-5-5"
        assert settings.llm_effort == "low"
        assert settings.llm_timeout_s == 10.0
        assert settings.llm_max_retries == 2
        assert settings.anthropic_api_key is None

    def test_llm_backend_without_key_is_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Choosing the ``llm`` backend without an API key fails at creation.

        Parameters
        ----------
        monkeypatch : pytest.MonkeyPatch
            Sets the environment variable for this test only.
        """
        monkeypatch.setenv("REVIEW_RIVER_CLASSIFIER_BACKEND", "llm")

        with pytest.raises(ValidationError):
            Settings()

    def test_llm_backend_with_key_hides_it(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With a key the ``llm`` backend is accepted, and the key never shows in ``repr``.

        Parameters
        ----------
        monkeypatch : pytest.MonkeyPatch
            Sets the environment variables for this test only.
        """
        monkeypatch.setenv("REVIEW_RIVER_CLASSIFIER_BACKEND", "llm")
        monkeypatch.setenv("REVIEW_RIVER_ANTHROPIC_API_KEY", "sk-ant-test-123")

        settings = Settings()

        assert settings.anthropic_api_key is not None
        assert settings.anthropic_api_key.get_secret_value() == "sk-ant-test-123"
        assert "sk-ant-test-123" not in repr(settings)
