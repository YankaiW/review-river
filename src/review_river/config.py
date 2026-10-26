"""Application settings, read from environment variables and an optional ``.env`` file."""

from typing import Literal, Self

from pydantic import Field, SecretStr, model_validator
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
    classifier_backend : {"keyword", "llm"}
        Which ``Classifier`` adapter to use. Env:
        ``REVIEW_RIVER_CLASSIFIER_BACKEND``.
    llm_model : str
        Claude model id for the LLM adapter. Env: ``REVIEW_RIVER_LLM_MODEL``.
    llm_effort : {"low", "medium", "high"}
        Reasoning effort per call. Env: ``REVIEW_RIVER_LLM_EFFORT``.
    llm_timeout_s : float
        Timeout per API attempt in seconds, must be positive. Env:
        ``REVIEW_RIVER_LLM_TIMEOUT_S``.
    llm_max_retries : int
        Retries the SDK makes on retryable errors. Env:
        ``REVIEW_RIVER_LLM_MAX_RETRIES``.
    anthropic_api_key : SecretStr or None
        Anthropic API key, hidden in ``repr``; read it with
        ``get_secret_value()``. Required when ``classifier_backend`` is
        ``"llm"``. Env: ``REVIEW_RIVER_ANTHROPIC_API_KEY``.

    Raises
    ------
    pydantic.ValidationError
        If an environment value fails validation, e.g. an unknown
        ``classifier_backend``, a non-positive ``llm_timeout_s``, or
        ``classifier_backend="llm"`` without an API key.
    """

    model_config = SettingsConfigDict(env_prefix="REVIEW_RIVER_", env_file=".env")

    app_name: str = "review-river"
    classifier_backend: Literal["keyword", "llm"] = "keyword"
    llm_model: str = "claude-opus-5-5"
    llm_effort: Literal["low", "medium", "high"] = "low"
    llm_timeout_s: float = Field(gt=0, default=10.0)
    llm_max_retries: int = Field(ge=0, default=2)
    anthropic_api_key: SecretStr | None = None

    @model_validator(mode="after")
    def _require_key_for_llm(self) -> Self:
        if self.classifier_backend == "llm" and not self.anthropic_api_key:
            raise ValueError("classifier_backend 'llm' needs REVIEW_RIVER_ANTHROPIC_API_KEY")
        return self
