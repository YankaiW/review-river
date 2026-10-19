"""Domain model for review-river.

Business concepts only: no FastAPI, database or LLM imports. Everything
else in the package depends on this module, never the other way round.
"""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class Category(StrEnum):
    """Topic a review is about, used to route it to the right team.

    Members are strings, so ``Category.BUG == "bug"`` and they serialise to
    JSON as plain strings.
    """

    BUG = "bug"
    PERFORMANCE = "performance"
    PAYMENTS = "payments"
    CONTENT = "content"
    OTHER = "other"


class Review(BaseModel):
    """A single piece of user feedback to classify.

    Attributes
    ----------
    text : str
        Review body. Surrounding whitespace is stripped before validation,
        so empty or whitespace-only text is rejected.

    Raises
    ------
    pydantic.ValidationError
        If ``text`` is empty after stripping.

    Examples
    --------
    >>> Review(text="  Game crashes on startup ").text
    'Game crashes on startup'
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    text: str = Field(min_length=1)


class Classification(BaseModel):
    """Result of classifying a review. Immutable value object.

    Attributes
    ----------
    category : Category
        Assigned topic. Plain strings such as ``"bug"`` are converted to the
        matching member; unknown values are rejected.
    confidence : float
        Classifier confidence in ``[0, 1]``.

    Raises
    ------
    pydantic.ValidationError
        If ``category`` is not a known value, ``confidence`` is outside
        ``[0, 1]``, or an attribute is assigned after creation.

    Examples
    --------
    >>> Classification(category="bug", confidence=0.6).category
    <Category.BUG: 'bug'>
    """

    model_config = ConfigDict(frozen=True)

    category: Category
    confidence: float = Field(ge=0.0, le=1.0)
