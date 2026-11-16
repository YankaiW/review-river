"""Probe Claude internals: tokens per language, latency, prompt cache.

An experiment, not product code: it calls the real API with the key from ``.env``.
Run one step at a time, e.g. ``uv run python scripts/probe_llm.py tokens``.
"""

import argparse
import asyncio

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


async def main() -> None:
    """Parse the step name and run it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=["tokens"])
    args = parser.parse_args()
    client = make_client()
    try:
        if args.step == "tokens":
            await probe_tokens(client=client)
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
