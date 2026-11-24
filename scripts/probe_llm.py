"""Probe Claude internals: tokens per language, latency, prompt cache.

An experiment, not product code: it calls the real API with the key from ``.env``.
Run one step at a time, e.g. ``uv run python scripts/probe_llm.py tokens``.
"""

import argparse
import asyncio
import logging
import random
import time
import uuid
from statistics import fmean

import numpy as np
from anthropic import APIConnectionError, APIStatusError, AsyncAnthropic
from anthropic.types import TextBlockParam, Usage

from review_river.classifier import SYSTEM_PROMPT
from review_river.deps import get_settings

MODEL = "claude-haiku-5-5"

REVIEWS = {
    "en": (
        "The game crashes every time I open the inventory since the last patch. "
        + "Please fix it, I can't play anymore."
    ),
    "de": (
        "Seit dem letzten Patch stürzt das Spiel jedes Mal ab, wenn ich das "
        + "Inventar öffne. Bitte behebt das, ich kann nicht mehr spielen."
    ),
    "zh": "自从上次补丁以后，每次打开背包游戏都会崩溃。请修复一下，我已经没法玩了。",  # noqa: RUF001
    "emoji": "Game crashes every time I open inventory since the patch 😡😡💥🎮 fix pls 🙏🙏🔥",
}

WORDS = (
    "game",
    "player",
    "patch",
    "server",
    "level",
    "quest",
    "update",
    "map",
    "weapon",
    "team",
    "match",
    "score",
    "story",
    "sound",
    "graphics",
    "price",
    "bug",
    "lag",
    "crash",
    "fun",
)

# Haiku 5.5, $ per MTok, prompts <= 100k tokens
PRICE_IN, PRICE_WRITE, PRICE_READ, PRICE_OUT = 0.10, 0.125, 0.01, 0.50

CACHE_WORDS = 32_000  # ~50K tokens, under the 100K price tier


def filler(n_words: int, seed: int) -> str:
    """Return ``n_words`` random common words, about one token each.

    A different ``seed`` gives a different text, so no two calls share a prefix.
    """
    rng = random.Random(seed)
    return " ".join(rng.choice(WORDS) for _ in range(n_words))


def fit_latency(rows: list[tuple[int, int, float]]) -> tuple[float, float, float]:
    """Fit ``latency_ms = a + b * input_tokens + c * output_tokens`` by least squares.

    Parameters
    ----------
    rows : list of (input_tokens, output_tokens, latency_ms)
        One row per call, with the token counts from ``usage``.

    Returns
    -------
    tuple of float
        ``a`` in ms, ``b`` and ``c`` in ms per token.
    """
    x = np.array([[1.0, i, o] for i, o, _ in rows])
    y = np.array([ms for _, _, ms in rows])
    coef = np.linalg.lstsq(x, y, rcond=None)[0]
    a, b, c = (float(v) for v in coef)
    return a, b, c


def call_cost(usage: Usage) -> float:
    """Return the $ cost of one call from its ``usage``."""
    write_cache = usage.cache_creation_input_tokens or 0
    read_cache = usage.cache_read_input_tokens or 0
    return (
        usage.input_tokens * PRICE_IN
        + write_cache * PRICE_WRITE
        + read_cache * PRICE_READ
        + usage.output_tokens * PRICE_OUT
    ) / 1e6


def make_client(max_retries: int = 0) -> AsyncAnthropic:
    """Build a client with a 120 s timeout and no retries by default.

    Measurements keep ``max_retries=0`` so a retry never distorts a latency.
    The overflow probe passes 2, the SDK default, to see whether the SDK
    retries an error.

    Parameters
    ----------
    max_retries : int
        Retries the SDK makes on retryable errors (408/409/429/5xx and
        connection errors, timeouts included).
    """
    key = get_settings().anthropic_api_key
    assert key is not None, "set REVIEW_RIVER_ANTHROPIC_API_KEY in .env"
    return AsyncAnthropic(api_key=key.get_secret_value(), timeout=120, max_retries=max_retries)


async def count_user(client: AsyncAnthropic, text: str) -> int:
    """Return the input tokens of one user message, with no system prompt."""
    resp = await client.messages.count_tokens(
        model=MODEL, messages=[{"role": "user", "content": text}]
    )
    return resp.input_tokens


async def probe_tokens(client: AsyncAnthropic) -> None:
    """Count tokens of the system prompt and of one review in four languages."""
    base = await count_user(client, "x")
    resp = await client.messages.count_tokens(
        model=MODEL,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": "x"}],
    )
    print(f"overhead: {base} tokens")
    print(f"system prompt: {resp.input_tokens - base} tokens (W2 logs: ~1474)")

    # English
    en = await count_user(client, REVIEWS["en"]) - base
    print(f"{'lang':<6} {'char':>5} {'tokens':>6} {'tok/char':>8} {'vs en':>6}")
    for lang, text in REVIEWS.items():
        tokens = await count_user(client, text) - base
        tok_char = tokens / len(text)
        rel_en = tokens / en
        print(f"{lang:<6} {len(text):>5} {tokens:>6} {tok_char:>8.2f} {rel_en:>6.2f}")


async def timed_call(
    client: AsyncAnthropic, prompt: str, max_tokens: int
) -> tuple[int, int, float]:
    """Send one message and time it end to end.

    Thinking is disabled, so the output holds no hidden thinking tokens and
    the token counts match what the latency depends on.

    Parameters
    ----------
    client : AsyncAnthropic
        Client built by ``make_client`` (no retries).
    prompt : str
        The whole user message.
    max_tokens : int
        Output cap; set it high enough that the answer is not cut off.

    Returns
    -------
    tuple of (int, int, float)
        ``input_tokens`` and ``output_tokens`` from ``usage``, and the wall
        time of the call in ms.
    """
    start = time.perf_counter()
    resp = await client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
        thinking={"type": "disabled"},
    )
    latency_ms = (time.perf_counter() - start) * 1000
    return (resp.usage.input_tokens, resp.usage.output_tokens, latency_ms)


async def probe_latency(client: AsyncAnthropic) -> None:
    """Measure how latency grows with input and with output length, and fit it.

    Runs a discarded warm-up call, then two sweeps on ``MODEL``, 3 calls per
    point, one at a time: input of about 1K, 10K and 50K tokens of filler
    with a ~1-token answer, then a short prompt asking for about 50, 200
    and 800 words. Prints each call, then the fit of
    ``latency_ms = a + b * input_tokens + c * output_tokens``.

    Parameters
    ----------
    client : AsyncAnthropic
        Client built by ``make_client`` (no retries).
    """
    # Warm-up: pays the connection setup; result thrown away
    await timed_call(client, "Reply with exactly: OK", max_tokens=5)

    rows: list[tuple[int, int, float]] = []
    seed = 0

    # Input sweep: long input, tiny answer
    for n in (1_000, 10_000, 50_000):
        for _ in range(3):
            seed += 1
            prompt = filler(n, seed) + "\n\nIgnore the text above. Reply with exactly: OK"
            row = await timed_call(client, prompt, max_tokens=5)
            rows.append(row)
            print(f"input n={n:>6} in={row[0]} out={row[1]} ms={row[2]:.0f}")

    # Output sweep: tiny input, long answer
    for w in (50, 200, 800):
        for _ in range(3):
            prompt = f"Write a story of about {w} words about a game update."
            row = await timed_call(client, prompt, max_tokens=2000)
            rows.append(row)
            print(f"output w={w:>5} in={row[0]} out={row[1]} ms={row[2]:.0f}")

    a, b, c = fit_latency(rows=rows)
    print(f"a = {a:.0f} ms")
    print(f"b = {b * 1000:.2f} ms per 1K input tokens")
    print(f"c = {c * 1000:.2f} ms per 1K output tokens")
    print(f"c / b = {c / b:.0f}")


async def cached_call(client: AsyncAnthropic, prefix: str, cache: bool) -> tuple[Usage, float]:
    """Send ``prefix`` as the system prompt with a fixed short question, and time it.

    Everything except ``prefix`` and ``cache`` is fixed (question, ``max_tokens``,
    thinking off), so calls differ only in whether the prefix is cached.

    Parameters
    ----------
    client : AsyncAnthropic
        Client built by ``make_client`` (no retries).
    prefix : str
        The whole system prompt. Start it with a nonce so a fresh prefix can
        never hit a cache entry left by an earlier run.
    cache : bool
        If True, put a 5-minute ``cache_control`` breakpoint at the end of the
        prefix: the first such call writes the cache, a repeat within 5 minutes
        reads it.

    Returns
    -------
    tuple of (Usage, float)
        The full ``usage`` (including ``cache_creation_input_tokens`` and
        ``cache_read_input_tokens``) and the wall time of the call in ms.
    """
    block: TextBlockParam = {"type": "text", "text": prefix}
    if cache:
        block["cache_control"] = {"type": "ephemeral"}
    start = time.perf_counter()
    resp = await client.messages.create(
        model=MODEL,
        max_tokens=5,
        system=[block],
        messages=[{"role": "user", "content": "Reply with exactly: OK"}],
        thinking={"type": "disabled"},
    )
    latency_ms = (time.perf_counter() - start) * 1000
    return resp.usage, latency_ms


def print_header() -> None:
    """Print the column names for ``print_row``, with the same widths."""
    print(f"{'label':<8} {'input':>6} {'write':>6} {'read':>6} {'out':>4} {'ms':>6} {'$':>9}")


def print_row(label: str, usage: Usage, ms: float) -> None:
    """Print one call: uncached input, cache write, cache read, output, ms and $.

    ``input`` is only the uncached remainder; the full prompt is
    ``input + write + read``.
    """
    write_tokens = usage.cache_creation_input_tokens or 0
    read_tokens = usage.cache_read_input_tokens or 0
    print(
        f"{label:<8} {usage.input_tokens:>6} {write_tokens:>6} {read_tokens:>6} "
        f"{usage.output_tokens:>4} {ms:>6.0f} {call_cost(usage):>9.5f}"
    )


async def probe_cache(client: AsyncAnthropic) -> None:
    """Compare an uncached prefix with a cache write and a cache read.

    Runs a discarded warm-up call, then 3 rounds on ``MODEL``, one call at a
    time. Each round builds two fresh ~50K-token prefixes (``CACHE_WORDS`` of
    filler, each starting with its own nonce) and sends:

    - ``control``: one prefix with no ``cache_control``, plain prefill;
    - ``write``: the other prefix with a breakpoint, which writes the cache;
    - ``read``: the same prefix again, which should read it.

    Prints each call, flags a ``read`` that missed the cache, then the mean
    latency and cost per label.

    Parameters
    ----------
    client : AsyncAnthropic
        Client built by ``make_client`` (no retries).
    """
    # 1. Warm-up: pays the connection setup; result thrown away
    await timed_call(client, "Reply with exactly: OK", max_tokens=5)

    # 2. Somewhere to collect results per label:
    results: dict[str, list[tuple[float, float]]] = {"control": [], "write": [], "read": []}

    print_header()
    for _ in range(3):
        # a. Two fresh prefixes, each starting with its own nonce:
        control_prefix = f"{uuid.uuid4().hex}\n" + filler(CACHE_WORDS, seed=0)
        cached_prefix = f"{uuid.uuid4().hex}\n" + filler(CACHE_WORDS, seed=1)

        # b. Three calls in order. For each one: await cached_call(...)
        for label, prefix, cache in (
            ("control", control_prefix, False),
            ("write", cached_prefix, True),
            ("read", cached_prefix, True),
        ):
            usage, ms = await cached_call(client, prefix, cache)
            print_row(label, usage, ms)
            if label == "read" and not usage.cache_read_input_tokens:
                print(" ^ cache MISS")
            results[label].append((ms, call_cost(usage)))

    print()
    print(f"{'mean':<8} {'ms':>6} {'$':>9}")
    for label, rows in results.items():
        ms_mean = fmean([r[0] for r in rows])
        cost_mean = fmean([r[1] for r in rows])
        print(f"{label:<8} {ms_mean:>6.0f} {cost_mean:>9.5f}")


async def probe_overflow(client: AsyncAnthropic) -> None:
    """Send a prompt about 10% over ``MODEL``'s context window and see what breaks.

    Reads the limit from the Models API, builds filler about 1.1 times that size
    and counts it with ``count_tokens`` (free). Stops if the prompt is not over
    the limit: a prompt that fits would be a real ~1M-token call, billed at
    Haiku's >100K-token rate (about $0.50). A rejected request is not billed.

    Parameters
    ----------
    client : AsyncAnthropic
        Client built by ``make_client``.
    """
    logging.basicConfig(level=logging.INFO)
    # Context window from the Models API, not a hard-coded number
    info = await client.models.retrieve(MODEL)
    limit = info.max_input_tokens
    assert limit is not None
    # Printed first, so it survives if count_tokens rejects the prompt
    print(f"limit: {limit:>9,} tokens")

    # Filler is ~1.55 tokens per word (latency sweep), so this is ~1.1 times the limit
    n_words = int(limit * 1.1 / 1.55)
    text = filler(n_words, seed=0)
    # Free, but it may itself refuse an over-limit prompt (prediction 4)
    user_input_tokens = await count_user(client, text)
    print(f"prompt: {user_input_tokens:>9,} tokens ({user_input_tokens / limit:.2f}x the limit)")
    # Guard: never send a prompt that fits, since that one would be billed
    if user_input_tokens <= limit:
        print("not over the limit, stopping")
        return

    start = time.perf_counter()
    try:
        await client.messages.create(
            model=MODEL,
            max_tokens=5,
            messages=[{"role": "user", "content": text}],
            thinking={"type": "disabled"},
        )
    except APIStatusError as e:
        ms = (time.perf_counter() - start) * 1000
        print(f"{type(e).__name__} {e.status_code} after {ms:.0f} ms")
        print(f"  message:    {e.message}")
        print(f"  request-id: {e.response.headers.get('request-id')}")
    except APIConnectionError as e:
        ms = (time.perf_counter() - start) * 1000
        print(f"{type(e).__name__} {ms:.0f} ms")
    else:
        print("NOT rejected: this call was billed")


async def main() -> None:
    """Parse the step name and run it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=["tokens", "latency", "cache", "overflow"])
    args = parser.parse_args()
    client = make_client(max_retries=2 if args.step == "overflow" else 0)
    try:
        if args.step == "tokens":
            await probe_tokens(client=client)
        elif args.step == "latency":
            await probe_latency(client=client)
        elif args.step == "cache":
            await probe_cache(client=client)
        elif args.step == "overflow":
            await probe_overflow(client=client)
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
