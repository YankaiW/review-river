"""Probe Claude internals: tokens per language, latency, prompt cache.

An experiment, not product code: it calls the real API with the key from ``.env``.
Run one step at a time, e.g. ``uv run python scripts/probe_llm.py tokens``.
"""

import argparse
import asyncio
import random
import time

import numpy as np
from anthropic import AsyncAnthropic

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


def make_client() -> AsyncAnthropic:
    """Build a client with no retries, so retries never distort a measurement."""
    key = get_settings().anthropic_api_key
    assert key is not None, "set REVIEW_RIVER_ANTHROPIC_API_KEY in .env"
    return AsyncAnthropic(api_key=key.get_secret_value(), timeout=120, max_retries=0)


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


async def main() -> None:
    """Parse the step name and run it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=["tokens", "latency"])
    args = parser.parse_args()
    client = make_client()
    try:
        if args.step == "tokens":
            await probe_tokens(client=client)
        elif args.step == "latency":
            await probe_latency(client=client)
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
