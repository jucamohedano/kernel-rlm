.PHONY: help install install-dev install-modal run-all \
        quickstart docker-repl lm-repl modal-repl \
        lint format format-check test check test-dashboard

help:
	@echo "RLM Examples Makefile"
	@echo ""
	@echo "Usage:"
	@echo "  make install        - Install base dependencies with uv"
	@echo "  make install-dev    - Install dev dependencies with uv"
	@echo "  make install-modal  - Install modal dependencies with uv"
	@echo "  make run-all        - Run all examples (requires all deps and API keys)"
	@echo ""
	@echo "Examples:"
	@echo "  make quickstart     - Run quickstart.py (needs OPENAI_API_KEY)"
	@echo "  make docker-repl    - Run docker_repl_example.py (needs Docker)"
	@echo "  make lm-repl        - Run lm_in_repl.py (needs PORTKEY_API_KEY)"
	@echo "  make modal-repl     - Run modal_repl_example.py (needs Modal)"
	@echo ""
	@echo "Development:"
	@echo "  make lint           - Check Ruff rules with the root pyproject.toml"
	@echo "  make format         - Format Python files with Ruff"
	@echo "  make format-check   - Check formatting without changing files"
	@echo "  make test           - Run tests"
	@echo "  make test-dashboard - Check dashboard JavaScript and run its tests (Node 22+)"
	@echo "  make check          - Run read-only lint, format, Python, and dashboard checks"

install:
	uv sync

install-dev:
	uv sync --group dev --group test

install-modal:
	uv pip install -e ".[modal]"

run-all: quickstart docker-repl lm-repl modal-repl

quickstart: install
	uv run python -m examples.quickstart

docker-repl: install
	uv run python -m examples.docker_repl_example

lm-repl: install
	uv run python -m examples.lm_in_repl

modal-repl: install-modal
	uv run python -m examples.modal_repl_example

lint:
	uv run --no-sync ruff check --config=pyproject.toml .

format:
	uv run --no-sync ruff format --config=pyproject.toml .

format-check:
	uv run --no-sync ruff format --check --config=pyproject.toml .

test:
	uv run --no-sync pytest

test-dashboard:
	node --check dashboard/app.js
	node --check dashboard/layout.js
	node --test tests/*dashboard*.test.cjs

check: lint format-check test test-dashboard
