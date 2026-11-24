# Learning notes

## Week 1

### 1. In FastAPI, when should an endpoint be `async def` and when plain `def`? What happens if you call a blocking function inside `async def`?

* When there is waiting time, e.g. I/O operation, while during the waiting time the function can work on other tasks, we should use `async def`.
* When there is no waiting time, the function do the tasks as sequential, we should use plain `def`.
* If calling a blocking function inside `async def`, it would break the whole pipeline.

**Reference answer:**
* `async def` when everything slow inside is awaited (async DB/HTTP/LLM clients). It runs on the event loop; each `await` hands the thread back, so one thread serves many waiting requests.
* Plain `def` when the endpoint calls blocking (sync-only) libraries: `requests`, `boto3`, `psycopg2`, `time.sleep`. FastAPI runs it in a thread pool (~40 threads), so the blocking stays in its own thread.
* Mixed: keep `async def` and wrap the one blocking call in `await asyncio.to_thread(fn, ...)`.
* A blocking call inside `async def` freezes the event loop, the only thread serving requests: every other request, even `/health`, waits.
* Evidence (5 concurrent requests, 2 s each): `async def` + `time.sleep` → 10 s; `async def` + `await asyncio.sleep` → 2 s; `def` + `time.sleep` → 2 s.


### 2. What does dependency injection (`Depends`) buy you in tests?

* It can reduce repeated code work.
* It can enforce dependencies before path operation functions and prepare input for it.
* It can confirm the input parameter data types and show in OpenAPI schema.

**Reference answer:**
* The point for tests: you can replace a real dependency with a fake, `app.dependency_overrides[get_classifier] = lambda: fake`, without changing any endpoint code.
* So API tests run in isolation: fast, deterministic, no network or LLM cost, and you choose the behaviour (a fixed answer, an error).
* A fake can also record what it received (`fake.seen`), proving what the endpoint passed on, and that invalid input never reached it.
* Evidence: `test_api.py` returns the fake's `content`/0.99 and asserts `fake.seen == []` on 422s.


### 3. Why define the classifier as a `Protocol` instead of calling a concrete class?

* When using `Protocal`, the endpoint/function could accept more classes with the required methods, otherwise, it would accept only that classifier class.

**Reference answer:**
* A `Protocol` is a contract by shape: any class with a matching `async classify(review) -> Classification` fits, no inheritance needed. pyright checks this before running.
* Callers depend on the contract, not an implementation (chapter 3), so adapters are swappable: keyword now, LLM in week 2, a fake in tests, a LoRA model in week 11, with no change to the API.
* Compared with a plain base class: subclassing silently inherits an empty `classify` if you misspell the method, giving `None` at runtime. With a Protocol, pyright flags the mismatch.


### 4. What does pyright strict catch that your tests don't, and the other way round?

* pyright catches the static type errors.
* tests check the logics inside functions/methods.

**Reference answer:**
* pyright: type and contract errors in every line, without running anything, including paths no test executes.
  Evidence: `add("1", 2)`; untyped parameters in strict mode; the misspelled `dependency_overides` (Python would silently create a new attribute); `KeywordClassifier` not fitting `Classifier`.
* Tests: runtime behaviour and logic that is type-correct but wrong.
  Evidence: `@dataclass` on `BaseModel` (pyright 0 errors, runtime crash); assigning to the frozen model (pyright missed it); `\bcrash\b` missing "crashes".
* Neither replaces the other: pyright proves the pieces fit; tests prove they do the right thing.


### 5. Why does a lockfile (`uv.lock`) plus a multi-stage Dockerfile make builds reproducible and small?

* uv.lock gives us a stable dependency environment.
* Two-stage Dockerfile split building into builder and runtime, where the runtime doesn't need the building env or cache at all.

**Reference answer:**
* Reproducible: `uv.lock` pins the exact version (and hash) of every package, including indirect dependencies. `uv sync --frozen` installs exactly that, and fails if the lockfile is out of date, so laptop, CI and Docker get identical environments.
* Small: the builder stage has uv and the build caches; the runtime stage copies only the finished `.venv` onto `python:3.12-slim`, with no dev tools, tests or source tree. Fewer files also means less attack surface.
* Fast rebuilds: dependencies are installed before the source is copied, so a code change reuses the cached dependency layer.
* Evidence: multi-stage ~160 MB vs a naive `python:3.12` + `COPY . .` image ~1.3 GB (measured 2026-10-20; record your own numbers).


## Week 2

### 1. What does structured output guarantee, and what can still go wrong when the JSON is valid?

* Structured output guarantees the pre-defined structure in output(mainly JSON with key-value pairs) that can be used in downstream task, where there is no unexpected keys.
* When JSON is valid, the value could be wrong still because LLM read value constrains just as description, the output value could be not in the defined range or rules.

**Reference answer:**
* Guaranteed: the API constrains generation to the JSON schema, so the text parses and has the required keys, the right types and only allowed enum values (`Category`). No missing keys, no free text around the JSON.
* Not guaranteed by the API: some schema constraints, e.g. numeric bounds. `confidence: Field(ge=0, le=1)` is checked locally by pydantic after the call.
* Still possible with valid JSON: the answer is wrong (wrong category, a confident but made-up `reason`), or `stop_reason` is `refusal`. A truncated answer (`max_tokens`) is the one case where the JSON is not valid at all.
* So check `stop_reason` and validate locally, and measure correctness with an eval: the schema guarantees shape, not truth.
* Evidence: "When new DLC would come out?" came back as valid JSON, `other` at 0.60, wrong under the label policy; `test_refusal_raises_output_error` rejects valid JSON with `stop_reason="refusal"`.


### 2. Which failures should be retried, by whom (the SDK or you), and which must not be?

* Connection errors, 408 Request Timeout, 409 Conflict, $29 Rate Limit and >=500 Internal errors should be retried by SDK.
* Others must not to be.

**Reference answer:**
* Retry, by the SDK: transient failures. Connection errors, timeouts, 408, 409, 429 and 5xx (including 529 overloaded), with exponential backoff that honours `retry-after`. Set `max_retries` (2) and a short `timeout` (10 s) and let the SDK do it.
* Not by you on top: a retry loop around the SDK multiplies attempts (3 × 3 = 9) and latency. Your layer reports the failure instead: 503 with `Retry-After`, so the caller retries later.
* Never retry: 400, 401, 403, 404 (e.g. unknown model), 413. They fail the same way every time, so a retry only adds latency.
* Output failures (`refusal`, `max_tokens`) are not retried either: the same request gives the same problem, and every attempt is billed. Fix the cause (prompt, `max_tokens`).
* Worst case before the client gets its 503: timeout × (retries + 1) + backoff sleeps ≈ 10 × 3 + ~1.5 ≈ 31.5 s.
* Evidence: `test_retries_overloaded_then_succeeds` (529 then 200, two requests); `test_bad_request_is_not_retried` (400, one request); break-it (b): a 401 is sent once.


### 3. Why create the LLM client once at startup instead of per request?

* We don't need re-create LLM client for each request, because requests can share one.
* Inside one shared client, memory and cached materials(e.g. prompts) can be shared, to improve efficiency and save cost.
* This can be done by lifespan, i.e. LLM client can be created before requests coming.
* One client for each request is a new start where there is no cached memory, and performance could be bad.

**Reference answer:**
* Connection pool: the shared client keeps TCP + TLS connections alive and reuses them, so only the first requests pay the handshake. A new client per request pays it every time.
* Bounded resources: one pool instead of one per request; per-request clients that are never closed leak sockets and memory under load.
* One owner: the lifespan builds it with the settings, stores it on `app.state` and closes it on shutdown.
* Not a reason: the prompt cache lives on Anthropic's servers (keyed by model and prompt prefix), and the Messages API is stateless. A new client per request still gets `cache_read`; the client holds no memory of earlier calls.
* Evidence: `test_api.py` asserts the client is open during requests and closed after shutdown; break-it (a), 50 concurrent requests on Haiku, warm burst: p50 1.56 s shared vs 1.67 s per request (~110 ms, the handshake), no errors in either.


### 4. What does one classification cost, and which tokens dominate the bill?

* No idea?
* Input context would dominate the bill.

**Reference answer:**
* Cost = tokens × price per million: input $4 / output $20 on Opus 5.5, $0.10 / $0.50 on Haiku 5.5. Output includes thinking tokens.
* Uncached Opus: ~1,500 in × $4 + ~60 out × $20 ≈ $0.007 per review ($7.25 per 1K). Input is ~84%, because the ~1,474-token system prompt repeats on every call.
* Cached Opus: 1,474 read × $0.20 + ~25 in × $4 + ~60 out × $20 ≈ $0.0017 ($1.97 per 1K). Output is now ~76%. The first call in 5 minutes pays the cache write at 1.25× input.
* Haiku: $0.19 → $0.06 per 1K, about 33× cheaper than Opus at the same accuracy on the 20 samples.
* Next levers after caching: a shorter `reason`, or the cheaper model.
* Evidence: smoke runs 2026-11-02 in `LEARNING_LOG.md`.


### 5. What does the model's self-reported `confidence` mean, and how would you test whether it is calibrated?

* `Confidence` means probability.
* We can validate it by backtest where there is real label, e.g. 80% is bug, then there should be 80% forecasted to be bug by LLM.

**Reference answer:**
* It is a number the model writes as text, a verbal estimate, not a probability read from the model's internals. It is not calibrated by default and moves with the prompt (`output_guidance` tells it what 0.9 and 0.5 mean).
* Calibrated means: among answers with confidence ≈ p, about p are correct. Matching class shares (predicted share of bugs = true share) is a different property.
* Test: a labelled set of hundreds of reviews; bin by confidence, plot accuracy against mean confidence per bin (reliability diagram), summarise with ECE or Brier score.
* If it is off: recalibrate on held-out data (isotonic or Platt), or use it only to rank, e.g. send the lowest-confidence reviews to a human.
* Evidence: 20 samples are far too few to measure it; Opus put 0.60 on the Mac review and was wrong. The W5 golden set is where this gets tested.


## Week 3

### 1. What is a token, and why does the same review cost a different number of tokens in English, German or Chinese?

* A token is a numeric value representing some plain word or a combination of words.
* Because the tokenizer is mostly based on English, where the most merging happens, when using other languages, the merging 
  cannot work well as English does, which give us longer encoded sequence.


### 2. Why do output tokens cost 5× input tokens, and why does a long prompt add little latency while a long answer adds a lot?

* During input prefill can do work on parallel, while the decoder during output needs to spend a whole pass of weights on each token.
* Because all input token can be processed in parallel, while one output token needs to be written one by one.


### 3. What does the KV cache do inside one request, and how does prompt caching across requests (W2) build on it?

* KV cache writes Key and Value matrices into cache, so when it process each token in input, they don't need to be re-calculated.
* The system prompt can be cached then each following requested prompt can re-use system prompt with lower cost.


### 4. If current Claude models take no `temperature`, where does run-to-run variation come from, and how much of it does your classifier show?

* The run-to-run variation can come from GPU math calculation.


### 5. For `/ask` over 10,000 reviews: how do putting everything in the context and retrieving the top 20 (RAG) differ in cost, latency and failure modes?

* Assumptions: 10,000 reviews × 80 tokens = 800K input for long context; top 20 × 80 + 50-token question ≈ 1.65K for RAG; 300-token answer either way. Prices from the pricing page, 2026-11-24.
* Cost per question:
  * Opus 5.5 (flat price up to 1M): long context $3.21 uncached, $0.17 cached (after the $4.01 first call), ~$0.20 averaged over 100 questions; RAG $0.013.
  * Haiku 5.5 (5× price for prompts over 100K, assumed to apply to the whole request, cache reads included): $0.40 uncached, $0.041 cached; RAG $0.0003.
  * Cached long context is still 13× (Opus) to 130× (Haiku) the cost of RAG. The cache only pays off while questions keep coming: the 5-minute TTL resets on each read.
* Latency, from the W3 Haiku fit: long context ~13.6 s uncached, ~7.6 s cached; RAG ~2.8 s plus retrieval. RAG time is mostly the 300-token answer (~1.9 s). The fit stops at ~77K tokens, so 800K may be slower.
* Failure modes:
  * Long context: reviews in the middle get used less reliably; ~12,500 reviews hit the 1M limit and the call fails with a 400 (W3 break-it); answers are hard to trace back to reviews.
  * RAG: retrieval misses the relevant reviews (low recall); the index lags new reviews; German or Chinese reviews may match poorly.
  * Both: counting questions ("how many mention crashes?") need classifier labels plus a database count, not the LLM.
* What W5 can measure: recall@20 on labelled questions; a planted review at start, middle and end of the long prompt; whether cited reviews support the answer (LLM judge); whether new reviews are found.
* Choose long context anyway when the corpus fits with margin and gets many questions within the cache TTL, when a question needs the whole corpus at once, or to prototype before building retrieval.

