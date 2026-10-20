"""Application settings, read from environment variables and an optional ``.env`` file."""

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration of review-river.

    Each field is read from an environment variable named ``REVIEW_RIVER_``
    plus the field name in upper case, then from ``.env``, then falls back to
    its default. Values are validated when ``Settings()`` is created, so an
    invalid value stops the app at startup.

    Attributes
    ----------
    app_name : str
        Name shown in the API docs. Env: ``REVIEW_RIVER_APP_NAME``.
    classifier_backend : {"keyword"}
        Which ``Classifier`` adapter to use. Env:
        ``REVIEW_RIVER_CLASSIFIER_BACKEND``.

    Raises
    ------
    pydantic.ValidationError
        If an environment value fails validation, e.g. an unknown
        ``classifier_backend``.
    """

    model_config = SettingsConfigDict(env_prefix="REVIEW_RIVER_", env_file=".env")

    app_name: str = "review-river"
    classifier_backend: Literal["keyword"] = "keyword"
