.PHONY: up seed down

up:
	docker compose up -d

seed:
	uv run python scripts/download_scifact.py

down:
	docker compose down
