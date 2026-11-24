"""Smoke run: classify the hand-labelled samples with each model and compare.

Prints accuracy, errors, cost per 1K reviews and p50 latency per model, then
the misclassified reviews. Calls the real API: needs the key in ``.env`` and
costs about $0.05 per run with the prompt cached. Run with ``make smoke``.

With ``--runs N`` the whole comparison repeats N times, then a stability
report shows how often each model gives the same label to the same review:
``uv run python scripts/smoke_llm.py --runs 3`` (about $0.05 per run).
"""

import argparse
import asyncio
import json
import logging
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
    predictions : list[tuple[Category, float] | None]
        Predicted category and confidence for every sample, in sample order;
        ``None`` where the call failed, so position ``i`` is the same review
        in every run.
    """

    model: str
    correct: int
    errors: int
    total: int
    cost_usd: float
    latencies_ms: list[float]
    misses: list[tuple[str, Category, Category]]  # text, gold, predicted
    predictions: list[tuple[Category, float] | None]  # per sample, in order; None = error


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
    predictions: list[tuple[Category, float] | None] = []
    try:
        for review, gold in samples:
            start_time = time.perf_counter()
            try:
                result_label = await llm_classifier.classify(review=review)
            except ClassifierUnavailable:
                errors += 1
                predictions.append(None)
                continue
            latencies_ms.append((time.perf_counter() - start_time) * 1000)
            predictions.append((result_label.category, result_label.confidence))
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
        predictions=predictions,
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


def print_stability(
    runs: dict[str, list[ModelResult]], samples: list[tuple[Review, Category]]
) -> None:
    """Print how stable each model's answers are across repeated runs.

    One line per model, then the unstable samples with the label from each
    run (``ERR`` for a failed call):

    - ``agree``: samples with the same label in every run; a failed call
      counts as a disagreement;
    - ``acc range``: lowest and highest accuracy across runs;
    - ``Δconf mean`` / ``Δconf max``: per sample, highest minus lowest
      confidence across runs, averaged and maximised over the samples where
      no call failed.

    Parameters
    ----------
    runs : dict[str, list[ModelResult]]
        Model id to its results, one per run, all over the same ``samples``.
    samples : list[tuple[Review, Category]]
        The reviews, in the order the runs classified them.
    """
    print(f"{'model':<18} {'agree':>6} {'acc range':>11} {'Δconf mean':>10} {'Δconf max':>9}")
    for model, results in runs.items():
        stable = 0
        spreads: list[float] = []
        unstable: list[tuple[str, list[str]]] = []  # text, label per run
        for i, (review, _) in enumerate(samples):
            preds = [r.predictions[i] for r in results]  # this sample, every run
            per_run = [str(p[0]) if p else "ERR" for p in preds]
            ok = [p for p in preds if p is not None]
            if "ERR" in per_run:
                unstable.append((review.text, per_run))
                continue
            labels = {p[0] for p in ok}
            confs = [p[1] for p in ok]
            spreads.append(max(confs) - min(confs))
            if len(labels) == 1:
                stable += 1
            else:
                unstable.append((review.text, per_run))

        accs = [r.correct / r.total for r in results]
        spreads_mean = statistics.fmean(spreads) if spreads else float("nan")
        spreads_max = max(spreads) if spreads else float("nan")

        agree = f"{stable}/{len(samples)}"
        acc_range = f"{min(accs):.0%}-{max(accs):.0%}"
        print(f"{model:<18} {agree:>6} {acc_range:>11} {spreads_mean:>10.2f} {spreads_max:>9.2f}")
        for text, per_run in unstable:
            print(" " + " / ".join(per_run) + " " + text)


async def main() -> None:
    """Run every model in ``MODELS`` over ``SAMPLES`` and print the comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=1, help="repeat the whole comparison N times")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be >= 1")
    logging.basicConfig(level=logging.INFO)
    # The classifier logs the model's reason at DEBUG; show it here for error analysis
    logging.getLogger("review_river.classifier").setLevel(logging.DEBUG)
    settings = get_settings()
    samples = load_samples(SAMPLES)
    runs: dict[str, list[ModelResult]] = {m: [] for m in MODELS}
    for i in range(args.runs):
        if args.runs > 1:
            print(f"\n=== run {i + 1}/{args.runs} ===")
        results = [
            await run_model(model=model, samples=samples, settings=settings) for model in MODELS
        ]
        print_table(results)
        for r in results:
            runs[r.model].append(r)

    if args.runs > 1:
        print_stability(runs, samples)


if __name__ == "__main__":
    asyncio.run(main())
