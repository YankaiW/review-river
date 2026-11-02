"""Dependency providers that FastAPI injects into endpoints via ``Depends``."""

from functools import lru_cache
from typing import Annotated

from fastapi import Depends, Request

from review_river.classifier import Classifier, KeywordClassifier, LLMClassifier, StructuredLLM
from review_river.config import Settings
from review_river.llm import LLMClient


@lru_cache
def get_settings() -> Settings:
    """Return the application settings, built once and cached.

    Returns
    -------
    Settings
        The same instance on every call. Call ``get_settings.cache_clear()``
        to force a re-read of the environment.
    """
    return Settings()


def build_llm(settings: Settings) -> LLMClient | None:
    """Build the shared LLM client when the backend needs one.

    Called once by the app lifespan, not per request. This is the only place
    the API key is unwrapped from its ``SecretStr``.

    Parameters
    ----------
    settings : Settings
        Application settings; the ``llm_*`` fields configure the client.

    Returns
    -------
    LLMClient or None
        A new client if ``classifier_backend`` is ``"llm"``, else ``None``.
    """
    if settings.classifier_backend == "llm":
        assert settings.anthropic_api_key is not None
        return LLMClient(
            api_key=settings.anthropic_api_key.get_secret_value(),
            model=settings.llm_model,
            effort=settings.llm_effort,
            timeout_s=settings.llm_timeout_s,
            max_retries=settings.llm_max_retries,
        )
    return None


def get_llm(request: Request) -> StructuredLLM | None:
    """Return the shared LLM client that the lifespan stored on ``app.state``.

    Parameters
    ----------
    request : Request
        Current request, injected by FastAPI; gives access to the app.

    Returns
    -------
    StructuredLLM or None
        The client, or ``None`` if the backend needs none or the lifespan
        has not run (e.g. ``TestClient`` used without ``with``).
    """
    return getattr(request.app.state, "llm", None)


def get_classifier(
    settings: Annotated[Settings, Depends(get_settings)],
    llm: Annotated[StructuredLLM | None, Depends(get_llm)],
) -> Classifier:
    """Return the classifier adapter selected by ``settings.classifier_backend``.

    Parameters
    ----------
    settings : Settings
        Application settings, injected by FastAPI from ``get_settings``.
    llm : StructuredLLM or None
        Shared LLM client, injected by FastAPI from ``get_llm``; used only by
        the ``"llm"`` backend.

    Returns
    -------
    Classifier
        An adapter that satisfies the ``Classifier`` port.

    Raises
    ------
    RuntimeError
        If the backend is ``"llm"`` but no client exists, a wiring bug.
    """
    match settings.classifier_backend:
        case "keyword":
            return KeywordClassifier()
        case "llm":
            if llm is None:
                raise RuntimeError("no LLM client: did the lifespan run?")
            return LLMClassifier(llm=llm)
