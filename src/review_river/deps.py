"""Dependency providers that FastAPI injects into endpoints via ``Depends``."""

from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from review_river.classifier import Classifier, KeywordClassifier
from review_river.config import Settings


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


def get_classifier(settings: Annotated[Settings, Depends(get_settings)]) -> Classifier:
    """Return the classifier adapter selected by ``settings.classifier_backend``.

    Parameters
    ----------
    settings : Settings
        Application settings, injected by FastAPI from ``get_settings``.

    Returns
    -------
    Classifier
        An adapter that satisfies the ``Classifier`` port.
    """
    match settings.classifier_backend:
        case "keyword":
            return KeywordClassifier()
