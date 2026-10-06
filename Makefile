.PHONY: up seed down test

up:
	docker compose up -d

seed:
	uv run python scripts/download_scifact.py

down:
	docker compose down

test:
	uv run pytest -q