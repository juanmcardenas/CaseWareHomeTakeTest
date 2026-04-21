.PHONY: install run test test-e2e migrate fmt clean

install:
	python -m venv .venv && . .venv/bin/activate && pip install -e .[dev]

run:
	PYTHONPATH=src uvicorn main:app --reload --host 0.0.0.0 --port 8000

test:
	PYTHONPATH=src pytest -v -m "not e2e"

test-e2e:
	PYTHONPATH=src pytest -v -m e2e

migrate:
	PYTHONPATH=src alembic upgrade head

migrate-create:
	PYTHONPATH=src alembic revision --autogenerate -m "$(MSG)"

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
