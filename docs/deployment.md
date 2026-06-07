# Delivery and Deployment

This project ships in two complementary modes, mirroring the "launch, monitor,
and maintain" stage of the end-to-end ML project workflow:

1. **Batch scoring** — run the pipeline end to end and write artifacts to disk.
2. **Online scoring** — serve the trained pipeline behind a small FastAPI app,
   packaged with Docker so a recruiter (or any teammate) can run it with one command.

## 1. Batch mode

```bash
python -m pip install -r requirements.txt
PYTHONPATH=src python -m airbnb_first_booking.cli train
```

Main outputs:

- `reports/metrics.json` — validation metrics (accuracy, balanced accuracy,
  macro F1/precision/recall and `ndcg_at_5`, the ranking metric used by the
  Kaggle leaderboard for this challenge).
- `models/model.joblib` — the fitted scikit-learn pipeline (preprocessing + model),
  serialized with `joblib` so it can be reloaded without retraining.

## 2. Online mode (FastAPI + Docker)

The same pipeline artifact is reused by a thin FastAPI service in
[`src/airbnb_first_booking/api.py`](../src/airbnb_first_booking/api.py),
exposing `/health` and `/predict` endpoints.

The fastest way to run it is Docker Compose — it builds the image, installs
dependencies, and serves the API on `http://localhost:8000`:

```bash
docker compose up --build
# or: make docker-run
```

Without Docker:

```bash
python -m pip install -r requirements.txt -r requirements-api.txt
PYTHONPATH=src uvicorn airbnb_first_booking.api:app --reload
```

Example request:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "records": [{
      "id": "abc123",
      "date_account_created": "2014-01-01",
      "timestamp_first_active": "20140101000000",
      "gender": "FEMALE",
      "age": 32,
      "signup_method": "basic",
      "signup_flow": 0,
      "language": "en",
      "affiliate_channel": "direct",
      "affiliate_provider": "direct",
      "first_affiliate_tracked": "untracked",
      "signup_app": "Web",
      "first_device_type": "Mac Desktop",
      "first_browser": "Chrome"
    }]
  }'
```

> The payload must include the same columns the model was trained on (the
> Kaggle `train_users_2.csv` schema). Missing columns raise a clear
> `ValueError` from the underlying `ColumnTransformer`.

The response includes the predicted destination class and, when the underlying
model exposes `predict_proba`, a confidence score per record.

## 3. Configuration and reproducibility

Every step (data paths, target, modeling hyperparameters, evaluation metrics)
is declared in [`configs/project.toml`](../configs/project.toml) and loaded by
[`config.py`](../src/airbnb_first_booking/config.py). This keeps the notebooks,
the CLI and the API aligned with a single source of truth, which makes the
pipeline reproducible end to end — train once, deploy the resulting artifact
anywhere.
