# review-river

[![CI](https://github.com/YankaiW/review-river/actions/workflows/ci.yml/badge.svg)](https://github.com/YankaiW/review-river/actions/workflows/ci.yml)

A customer-feedback assistant for game reviews. It will do two jobs, and
prove with evals how well it does them:

1. **Classify and route**: each Steam review gets a category (bug,
   performance, payments, content, other) and a confidence score, so it
   reaches the right team.
2. **Answer questions**: "What do players complain about since the last
   patch?" gets an answer that cites the actual reviews (RAG).

**Status:** week 1 of 13. The service skeleton is in place: a typed domain
model, a swappable `Classifier` port with a keyword-based adapter, a FastAPI
API, tests, a Docker image and CI. The LLM classifier, retrieval and evals
come next (see [Roadmap](#roadmap)).

## Architecture

```
HTTP  ──►  api.py  ──►  Classifier (port)  ──►  KeywordClassifier   (now)
          (thin)          ▲                     LLM classifier      (week 2)
                          │                     fine-tuned model    (week 11)
                     deps.py: picks the adapter from settings
                          │
                     domain.py: Review, Category, Classification
                     (business rules, no framework imports)
```

Endpoints depend only on the `Classifier` protocol. Adapters are injected via
FastAPI's `Depends`, so tests swap in a fake and production swaps in an LLM
without touching the API code.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) (it installs Python 3.12 if needed).

```bash
make install   # dependencies + pre-commit hooks
make run       # API with auto-reload on http://127.0.0.1:8000
```

Interactive API docs: <http://127.0.0.1:8000/docs>

```bash
curl -s -X POST http://127.0.0.1:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "Game crashes on startup"}'
# {"category":"bug","confidence":0.6}
```

### With Docker

```bash
make docker-build
make docker-run
```

Multi-stage image: dependencies are installed with uv in a builder stage; the
runtime stage contains only the virtualenv and runs as a non-root user with a
health check on `/health`.

## Configuration

Settings are read from environment variables (or a local `.env` file):

| Variable | Default | Meaning |
|---|---|---|
| `REVIEW_RIVER_APP_NAME` | `review-river` | Title shown in the API docs |
| `REVIEW_RIVER_CLASSIFIER_BACKEND` | `keyword` | Which classifier adapter to use |

An invalid value stops the app at startup rather than failing on a request.

## Development

```bash
make check     # what CI runs: ruff lint + format check, pyright (strict), pytest
make format    # reformat and apply auto-fixes
make help      # all targets
```

Pre-commit runs ruff and pyright on every commit. CI runs `make check` and
builds the Docker image on every pull request and push to `main`.

```
src/review_river/
  domain.py      Review, Category, Classification
  classifier.py  Classifier port + KeywordClassifier adapter
  config.py      Settings from environment variables
  deps.py        Dependency providers for FastAPI
  api.py         FastAPI app: GET /health, POST /classify
tests/           unit tests per module + API tests with an injected fake
```

## Roadmap

| Week | Adds |
|---|---|
| 2 | LLM classifier adapter: structured output, retries, token and cost logging |
| 4 | Ingest Steam reviews; embeddings, pgvector, hybrid search; `/ask` endpoint |
| 5–6 | Evals: hand-labelled golden sets, LLM judge calibrated against them, eval gate in CI, tracing |
| 10 | Deployment, caching, cost dashboard, prompt-injection defences |
| 11 | Fine-tuning experiment: embeddings + logistic regression vs prompted LLM vs LoRA |
