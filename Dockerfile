FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app

COPY requirements.txt requirements-api.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt -r requirements-api.txt

COPY configs ./configs
COPY src ./src
COPY models ./models

EXPOSE 8000

CMD ["uvicorn", "airbnb_first_booking.api:app", "--host", "0.0.0.0", "--port", "8000"]
