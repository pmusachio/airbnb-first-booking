.PHONY: install download-data profile train analyze test api docker-build docker-run docker-stop

install:
	python -m pip install -r requirements.txt -r requirements-api.txt

download-data:
	bash scripts/download_data.sh

profile:
	PYTHONPATH=src python -m airbnb_first_booking.cli profile

train:
	PYTHONPATH=src python -m airbnb_first_booking.cli train

analyze:
	PYTHONPATH=src python -m airbnb_first_booking.cli analyze

test:
	python -m pytest

api:
	PYTHONPATH=src uvicorn airbnb_first_booking.api:app --reload

docker-build:
	docker compose build

docker-run:
	docker compose up --build

docker-stop:
	docker compose down
