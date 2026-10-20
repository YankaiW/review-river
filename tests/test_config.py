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
