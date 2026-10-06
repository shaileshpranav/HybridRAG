.PHONY: up data seed down

up:
	docker compose up -d

data:
	uv run python scripts/download_scifact.py

seed:
	uv run python scripts/seed.py

down:
	docker compose down
