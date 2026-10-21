.DEFAULT_GOAL := help
.PHONY: help install format lint typecheck test check run clean docker-build docker-run

help:  ## Show this list
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  make %-13s %s\n", $$1, $$2}'

install:  ## Install dependencies and the pre-commit hooks
	uv sync
	uv run pre-commit install

format:  ## Reformat code and apply auto-fixable lint fixes
	uv run ruff format .
	uv run ruff check --fix .

lint:  ## Check lint and formatting without changing files
	uv run ruff check .
	uv run ruff format --check .

typecheck:  ## Run pyright (strict)
	uv run pyright

test:  ## Run the test suite
	uv run pytest -v

check: lint typecheck test  ## Everything CI runs: lint, typecheck, test

run:  ## Start the API with auto-reload on http://127.0.0.1:8000
	uv run uvicorn review_river.api:app --reload

clean:  ## Remove caches
	find . -type d -name __pycache__ -not -path './.venv/*' -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache

docker-build:  ## Build the Docker image
	docker build -t review-river .

docker-run:  ## Run the image on http://127.0.0.1:8000
	docker run --rm -p 8000:8000 review-river
