"""review-river: classify game reviews and answer questions about them.

A FastAPI service that routes each Steam review to the right team by
category, and (from week 4) answers questions about reviews with cited
sources. Business rules live in ``domain``; everything else depends on it.

Modules
-------
api
    FastAPI app with the ``/health`` and ``/classify`` endpoints.
domain
    Business concepts: ``Review``, ``Category``, ``Classification``.
classifier
    The ``Classifier`` port and its adapters.
config
    ``Settings`` read from environment variables.
deps
    Dependency providers that FastAPI injects into endpoints.
"""
