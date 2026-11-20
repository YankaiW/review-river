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

Latency model, Haiku 5.5, thinking off, no cache, sequential calls, 3 per point (means; the 1.5K point drops one call that returned 0 tokens):

| sweep | input tokens | output tokens | latency ms |
|---|---|---|---|
| input | ~1,570 | 4 | 1,213 |
| input | ~15,510 | 4 | 1,240 |
| input | ~77,380 | 4 | 1,881 |
| output | 21 | ~83 | 1,116 |
| output | 21 | ~440 | 3,525 |
| output | 21 | ~1,266 | 9,062 |

Fit `latency_ms ≈ a + b·input + c·output` (all 18 calls / without the 0-token call):

- `a` ≈ 800 / 849 ms fixed per call; `b` ≈ 14.2 / 13.5 ms per 1K input tokens; `c` ≈ 6.47 / 6.42 ms per output token (~155 tokens/s).
- `c / b` ≈ 456 / 477: in time, one output token costs about as much as ~460 input tokens. Prediction: _(add yours)_.
- Price ratio is only 5×. Prefill keeps the GPU busy but runs in parallel, so it is short in wall time; each decode step serves many requests batched together. You pay for GPU work, you wait for the sequence of steps.
- Check against W2: a classification (~1.6K in, ~60 out) predicts ~1.2 s; the W2 Haiku p50 was 1,363 ms (with some thinking).
- The line fits loosely: the input-sweep calls sit ~+100 to +340 ms above it, the 50-word calls ~−250 ms below. Possible causes: the input sweep ran first (no shuffle), or time to first token is not linear in input. Shuffled calls or streamed TTFT would separate them.
- One call returned 0 output tokens in 442 ms; `stop_reason` was not recorded.

Prompt cache vs no cache, Haiku 5.5, a ~49.7K-token system prefix (nonce first, so each round starts cold), thinking off, 3 rounds of control → write → read, sequential:

| call | uncached input | cache write | cache read | ms mean / median | $ mean | vs control $ |
|---|---|---|---|---|---|---|
| control (no `cache_control`) | ~49,500 | 0 | 0 | 1,506 / 1,539 | 0.00495 | 1× |
| write (breakpoint, cold) | 16 | ~49,665 | 0 | 1,627 / 1,409 | 0.00621 | 1.25× |
| read (same prefix again) | 16 | 0 | ~49,665 | 1,183 / 1,134 | 0.00050 | 0.10× |

- The cache hit in all 3 rounds. `input_tokens` is only the uncached remainder (the 16-token question); the full prompt is input + write + read.
- Cost matches the price list exactly: a write is 1.25× and a read 0.10× of the uncached call, so the cache pays for itself on the first read (1.35× vs 2× for two calls).
- Control matches the latency fit: a + b·49.5K + c·4 ≈ 1,543 ms predicted vs 1,506 measured.
- A read saves ~320 ms (mean) / ~400 ms (median), about half of the ~670 ms prefill the fit predicts. A fully skipped prefill would give ≈ a ≈ 875 ms; the read takes 1,183 ms. The prompt is still uploaded and tokenized, and the stored KV for 50K tokens has to be loaded back. So a cached token costs ~half the time but a tenth of the price of an uncached one: caching is mainly a cost lever.
- A write costs no extra time (same prefill, KV kept). The +120 ms mean comes from one 2,073 ms outlier; by median the write is faster than control. n = 3.
- For the 1.5K-token classifier system prompt this means at most ~20 ms saved out of ~1.2 s, but those tokens get 10× cheaper (check-yourself Q5).
- Prediction: _(add yours)_.

Run-to-run stability, `smoke_llm.py --runs 3`: 20 labelled reviews × 3 runs per model, sequential, prompt cached, default settings (10 s timeout, 2 retries):

| model | accuracy per run | errors per run | agree (same label in all 3) | Δconf mean / max | $ per 1K reviews |
|---|---|---|---|---|---|
| Opus 5.5 | 95% / 90% / 95% | 1 / 2 / 1 | 18/20 | 0.01 / 0.10 | 1.97 / 1.51 / 1.61 |
| Haiku 5.5 | 100% / 100% / 100% | 0 / 0 / 0 | 20/20 | 0.05 / 0.25 | 0.06 / 0.05 / 0.05 |

- No label flips: every call that answered gave the same label in all 3 runs, on both models, and every answer was correct. Opus's 18/20 and 90–95% come only from failed calls ("frame rate on Mac": ERR / ERR / performance; "What a game!": other / ERR / ERR). `agree` mixes flips and errors; splitting them would show 0 flips, 4 errors.
- Sampling can't be set on these models, but on a clear label policy it doesn't change the answer. The variation shows up in confidence, and only on reviews already rated low: Opus "cannot be played on Mac" 0.70 / 0.60 / 0.70, Haiku "laptop is very hot" 0.60 / 0.45 / 0.70. Confident answers (≥ 0.9) barely move. Confidence comes in 0.05 steps: a number the model writes, stable enough to rank by, not a calibrated probability.
- The errors are most likely timeouts: the failed attempts got no HTTP response (httpx logs one for every 429/5xx), and the one retry that succeeded took 17.4 s ≈ a 10 s timeout + backoff + a ~7 s second attempt. Opus's typical call takes ~1.6 s but its slow tail reaches 7–17 s; Haiku's slowest was 2.2 s. A failed review waits ~30 s (3 attempts × 10 s) before `ClassifierUnavailable`. Not confirmed: `classifier.py` wraps the `APIError` and the smoke script never prints `e.__cause__` (Session 4 debt).
- Open for Session 4: at 10 s, ~2–3% of Opus calls fail here after a 30 s wait. Options: a longer timeout, fail fast with fewer retries, or fall back to Haiku on a timeout.
- Cost: run 1 includes the cache writes; ~$0.11 for all 120 calls.
- Prediction: _(add yours)_.

Break-it: a prompt over the context window, Haiku 5.5, `probe_llm.py overflow` then `/classify` (`classifier_backend=llm`). Free: a rejected request is not billed.

| layer | what happened |
|---|---|
| Models API | `max_input_tokens` = 1,000,000 |
| `count_tokens` | returned 1,100,373 (1.10× the limit), no error |
| `messages.create`, SDK `max_retries=2` | `400 BadRequestError`, `invalid_request_error`: "prompt is too long: 1100396 tokens > 1000000 maximum", after 933 ms; one attempt, no retry |
| `POST /classify` (service: 10 s timeout, 2 retries) | `503 Service Unavailable`, `Retry-After: 5`, `{"detail": "classifier unavailable, try again later"}`, after 1.58 s |

- `count_tokens` counts, it doesn't check: it returns the size of an over-limit prompt without an error, so a pre-check has to compare against `max_input_tokens` itself. Its count was 23 tokens under the real one (1,100,373 vs 1,100,396), so it is an estimate here too (same as the system prompt in Piece 1).
- The API enforces the limit by token count and answers fast: ~4.5 MB uploaded and 1.1M tokens counted in under 1 s, faster than a normal 1.5K-token classification, because nothing is generated.
- The SDK does not retry a 400: it only retries 408/409/429/5xx and connection errors, since a bad request fails the same way every time.
- The service gets it wrong. `Review` has no length limit, so the whole text goes to Claude; `classifier.py` wraps every `APIError` (the 400 included) in `ClassifierUnavailable`, and `api.py` turns that into 503 + `Retry-After: 5`. A client that obeys it retries an input that can never succeed, uploading 4.5 MB each time; monitoring sees an outage; the caller never learns the input is the problem (the cause is only in the server log).
- Session 4: (1) cap `Review.text` (the samples are all < 200 chars; ~10–20K chars leaves room) so FastAPI returns 422 before any provider call; (2) map provider errors by cause: an input the provider rejects → 4xx to the caller, a wrong key/model/parameter (401/403/404/400) → 500, only 429/5xx/timeouts → 503 with `Retry-After`.
- Prediction: _(add yours)_.
