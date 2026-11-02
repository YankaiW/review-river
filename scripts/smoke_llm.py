"""Smoke run: classify the hand-labelled samples with each model and compare.

Prints accuracy, errors, cost per 1K reviews and p50 latency per model, then
the misclassified reviews. Calls the real API: needs the key in ``.env`` and
costs about $0.15 per run. Run with ``make smoke``.
"""

import asyncio
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

from review_river.classifier import ClassifierUnavailable, LLMClassifier
from review_river.config import Settings
from review_river.deps import get_settings
from review_river.domain import Category, Review
from review_river.llm import LLMClient

MODELS = ("claude-opus-5-5", "claude-haiku-5-5")
SAMPLES = Path(__file__).parent / "samples.jsonl"


@dataclass
class ModelResult:
    """Outcome of one model over all samples.

    Attributes
    ----------
    model : str
        Claude model id.
    correct : int
        Samples whose predicted category matches the label.
    errors : int
        Samples where the classifier raised ``ClassifierUnavailable``.
    total : int
        Number of samples; ``correct + len(misses) + errors == total``.
    cost_usd : float
        Billed cost of all calls, from ``LLMClient.total_cost_usd``.
    latencies_ms : list[float]
        Wall time of each successful ``classify`` call, retries included.
    misses : list[tuple[str, Category, Category]]
        Review text, label and predicted category of each wrong answer.
    """

    model: str
    correct: int
    errors: int
    total: int
    cost_usd: float
    latencies_ms: list[float]
    misses: list[tuple[str, Category, Category]]  # text, gold, predicted


def load_samples(path: Path) -> list[tuple[Review, Category]]:
    """Read labelled reviews from a JSONL file.

    Parameters
    ----------
    path : Path
        File with one ``{"text": ..., "label": ...}`` object per line; blank
        lines are skipped.

    Returns
    -------
    list[tuple[Review, Category]]
        Each review with its hand-assigned category.
    """
    samples: list[tuple[Review, Category]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        samples.append((Review(text=d["text"]), Category(d["label"])))

    return samples


async def run_model(
    model: str, samples: list[tuple[Review, Category]], settings: Settings
) -> ModelResult:
    """Classify every sample with one model, one call at a time.

    Calls run sequentially so latency is not distorted by queueing or rate
    limits. A failed call counts as an error and the run continues.

    Parameters
    ----------
    model : str
        Claude model id; overrides ``settings.llm_model``.
    samples : list[tuple[Review, Category]]
        Reviews with their labels.
    settings : Settings
        Source of the API key, effort, timeout and retries.

    Returns
    -------
    ModelResult
        Counts, cost, latencies and misses for this model.
    """
    assert settings.anthropic_api_key is not None
    llm_client = LLMClient(
        api_key=settings.anthropic_api_key.get_secret_value(),
        model=model,
        effort=settings.llm_effort,
        timeout_s=settings.llm_timeout_s,
        max_retries=settings.llm_max_retries,
    )
    llm_classifier = LLMClassifier(llm=llm_client)
    correct = 0
    misses: list[tuple[str, Category, Category]] = []
    errors = 0
    latencies_ms: list[float] = []
    try:
        for review, gold in samples:
            start_time = time.perf_counter()
            try:
                result_label = await llm_classifier.classify(review=review)
            except ClassifierUnavailable:
                errors += 1
                continue
            latencies_ms.append((time.perf_counter() - start_time) * 1000)
            if result_label.category == gold:
                correct += 1
            else:
                misses.append((review.text, gold, result_label.category))
    finally:
        await llm_client.close()

    return ModelResult(
        model=model,
        correct=correct,
        errors=errors,
        total=len(samples),
        cost_usd=llm_client.total_cost_usd,
        latencies_ms=latencies_ms,
        misses=misses,
    )


def print_table(results: list[ModelResult]) -> None:
    """Print one row per model, then the misclassified reviews.

    Parameters
    ----------
    results : list[ModelResult]
        One result per model, in the order to print.
    """
    print(f"{'model':<18} {'accuracy':>8} {'errors':>6} {'$ per 1K':>9} {'p50 ms':>7}")
    for r in results:
        p50 = statistics.median(r.latencies_ms) if r.latencies_ms else float("nan")
        cost_per_1k = r.cost_usd / r.total * 1000
        print(
            f"{r.model:<18} {r.correct / r.total:>8.0%} {r.errors:>6} "
            f"{cost_per_1k:>9.2f} {p50:>7.0f}"
        )
    for r in results:
        if r.misses:
            print(f"\nMisses, {r.model}:")
            for text, gold, predicted in r.misses:
                print(f"  label={gold} predicted={predicted}  {text}")


async def main() -> None:
    """Run every model in ``MODELS`` over ``SAMPLES`` and print the comparison."""
    settings = get_settings()
    samples = load_samples(SAMPLES)
    results = [await run_model(model=model, samples=samples, settings=settings) for model in MODELS]
    print_table(results)


if __name__ == "__main__":
    asyncio.run(main())
