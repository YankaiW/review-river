# Learning log

## W1: FastAPI service, keyword classifier, Docker, CI

- **Built:** domain models, `Classifier` port with a keyword adapter, `/health` and `/classify`, multi-stage Dockerfile, CI running `make check` and the Docker build.
- **Surprised:** 5 concurrent requests, 2 s of sleep each: `async def` + `time.sleep` → 10 s; `async def` + `await asyncio.sleep` → 2 s; `def` + `time.sleep` → 2 s.
- **Broke:** a blocking call inside `async def` freezes the event loop, so every other request waits, `/health` included.
- **Open question:** ?
- **Time:** 1 week

## W2: LLM call layer (2026-11-02)

- **Built:** `LLMClient` (structured output, typed errors, one log line per call with tokens, cost and request id), `LLMClassifier` behind the same port, a shared client via the lifespan, 503 on provider failure, a smoke script over 20 hand-labelled reviews.
- **Surprised:** before caching, input was ~84% of Opus's cost per call because a ~1,474-token prompt prefix repeats on every call; after caching, output (thinking plus `reason`) is ~76%. Haiku matched Opus at 1/33 of the cost.
- **Broke:** `_cost_usd` billed cache reads at `price_in × 0.20` instead of $0.20/MTok (caught before a cached run). A top-level `cache_control` would have put the breakpoint on the review, which changes every call, so nothing would be reused.
- **Open question:** the 20 samples shaped the prompt rules, so 100% is training accuracy. How well does it do on reviews it has never seen (W4 Steam data), and is `confidence` calibrated?
- **Time:** 8 days

Smoke run, 20 samples, effort `low`, sequential calls:

| run | model | accuracy | $ per 1K | p50 ms |
|---|---|---|---|---|
| baseline | Opus 5.5 | 90% | 7.25 | ~2,000 |
| baseline | Haiku 5.5 | 95% | 0.19 | ~1,500 |
| + cached system prompt | Opus 5.5 | 90% | 1.97 | 1,715 |
| + cached system prompt | Haiku 5.5 | 95% | 0.06 | 1,422 |
| + label policy rules | Opus 5.5 | 100% | 2.02 | 2,174 |
| + label policy rules | Haiku 5.5 | 100% | 0.06 | 1,363 |

- Caching: call 1 writes ~1,474 tokens at 1.25× the input price, calls 2–20 read them at 0.05× (Opus) or 0.1× (Haiku). Below 512 tokens nothing is cached.
- Misses before the rules were label-policy gaps, not model errors: a question about upcoming DLC and "cannot be played on Mac". Fixed by routing on topic rather than tone, and calling platform failures bugs.
- At n=20 one review is 5 points of accuracy, so accuracy cannot separate the models; cost and latency can.

Break-its, failure paths:

- Wrong API key: the client gets 503 with `Retry-After: 5`; the 401 is not retried. The app starts fine with a bad key, since only its presence is checked.
- `max_tokens=16`: 503, and no `llm_call` line. The JSON was cut off, so `messages.parse` raised before any usage was read: a billed call that is neither logged nor counted in `total_cost_usd`.

Break-it, shared client vs a new client per request: 50 concurrent `/classify` requests on Haiku 5.5, second (warm) burst:

| client | p50 | statuses |
|---|---|---|
| shared, built in the lifespan | 1.56 s | 50 × 200 |
| new per request, never closed | 1.67 s | 50 × 200 |

- ~110 ms (7%) more per request: a new TCP + TLS handshake and HTTP client every time. Small next to model latency; the stronger case for one client is bounded sockets and no unclosed clients piling up. One run per arm, so the gap is indicative.

## W3: Just enough internals (in progress)

Tokens per language and the system prompt, Haiku 5.5, `count_tokens` (free), a 9-token message overhead subtracted:

| text | chars | tokens | tokens/char | vs English |
|---|---|---|---|---|
| English review | 107 | 32 | 0.30 | 1.00 |
| German, same review | 129 | 65 | 0.50 | 2.03 |
| Chinese, same review | 36 | 40 | 1.11 | 1.25 |
| English + 6 emoji | 73 | 40 | 0.55 | 1.25 |

- Tokens per character misleads across scripts: Chinese has the most per character, but German costs the most for the same content (longer sentence, words split into more pieces). Compare tokens for the same content. One sentence per language, so the ratios are indicative.
- System prompt: 959 tokens alone, 1,591 with the output schema, so the schema adds 632 (~40% of the prompt). `transform_schema(LLMLabel)` shows why: the `LLMLabel` and `Category` docstrings are sent as schema descriptions, so the model reads them on every call, and editing one changes the prompt and invalidates the cache.
- 1,591 counted vs ~1,474 billed (W2 cache writes): `count_tokens` is an estimate; billed `usage` is the ground truth.
- Opus 5.5 gives identical counts on these texts, so the two models tokenize them the same way.
