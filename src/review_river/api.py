"""HTTP entrypoint: FastAPI app exposing health and classification endpoints."""

from typing import Annotated

from fastapi import Depends, FastAPI

from review_river.classifier import Classifier
from review_river.deps import get_classifier, get_settings
from review_river.domain import Classification, Review

ClassifierDep = Annotated[Classifier, Depends(get_classifier)]

app = FastAPI(title=get_settings().app_name)


@app.get("/health")
async def heath() -> dict[str, str]:
    """Report that the service is up.

    Used by Docker and load balancers to check liveness. Does not touch the
    classifier or any other dependency.

    Returns
    -------
    dict[str, str]
        Always ``{"status": "ok"}``.
    """
    return {"status": "ok"}


@app.post("/classify")
async def classify(review: Review, classifier: ClassifierDep) -> Classification:
    """Classify one review into a category with a confidence score.

    The request body is validated against ``Review`` before this runs, so
    empty or whitespace-only text is rejected with HTTP 422.

    Parameters
    ----------
    review : Review
        The review to classify, parsed from the JSON request body.
    classifier : Classifier
        Adapter chosen by ``get_classifier``, injected by FastAPI.

    Returns
    -------
    Classification
        Assigned category and confidence, serialised as JSON.
    """
    return await classifier.classify(review=review)
