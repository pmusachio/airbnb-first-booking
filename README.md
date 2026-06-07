# Airbnb Predict First Booking

A data science portfolio project built on the Kaggle challenge
[Airbnb Recruiting New User Bookings](https://www.kaggle.com/competitions/airbnb-recruiting-new-user-bookings).

It follows the end-to-end machine learning project checklist popularized by
*Hands-On Machine Learning with Scikit-Learn and PyTorch* (Aurélien Géron):
frame the business problem, explore and prepare the data, train and evaluate
models, translate results into business value, and ship a reproducible,
deployable solution — not just a notebook.

> **For recruiters / reviewers:** jump straight to [section 3, "How to Run This Project"](#3-how-to-run-this-project) for a copy-paste path to a working API in under five minutes.

## 1. Business Problem

Airbnb wants to anticipate the first destination country of new users so it
can personalize communication, marketing campaigns and recommendations from
the very start of the user journey.

**Goal:** predict the first booking destination of new Airbnb users and
produce a solution aligned with the Kaggle submission format (a ranked top-5
list of destinations per user).

**Primary metrics:** NDCG@5 (the official competition metric, implemented in
[`models.py`](src/airbnb_first_booking/models.py)), macro F1 and balanced accuracy.

## 2. Business Assumptions

- The first booking is shaped by signup data, acquisition channel and session behavior.
- The `NDF` class ("no destination found", i.e. no booking) is a relevant outcome on its own, not noise to discard.
- Because the business action can recommend more than one destination, evaluation must reward good top-N rankings — hence NDCG@5 as the headline metric.

## 3. How to Run This Project

The fastest path to a working prediction API — no notebook required:

```bash
git clone <this-repository-url> airbnb-first-booking
cd airbnb-first-booking
docker compose up --build
```

Then, in another terminal:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "records": [{
      "id": "demo-1",
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

You should get back a predicted destination class (and a confidence score, if
the loaded model supports `predict_proba`). The `/health` endpoint is also
available for a quick smoke test: `curl http://localhost:8000/health`.

A pre-trained pipeline ships in [`models/model.joblib`](models/model.joblib),
so the container starts ready to score — no training step needed to try it out.

### Running without Docker

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt -r requirements-api.txt
PYTHONPATH=src uvicorn airbnb_first_booking.api:app --reload
```

### Reproducing the full pipeline (download → train → serve)

The raw competition data is **not** stored in this repository (see
[section 5](#5-data-source)) — it is fetched on demand from Kaggle:

```bash
python -m pip install -r requirements.txt

# 1. Download the dataset from Kaggle into data/raw/ (requires a Kaggle API token)
make download-data                                                  # or: bash scripts/download_data.sh

# 2. Run the pipeline
PYTHONPATH=src python -m airbnb_first_booking.cli validate-config   # sanity-check configs/project.toml
PYTHONPATH=src python -m airbnb_first_booking.cli profile           # writes reports/data_profile.json
PYTHONPATH=src python -m airbnb_first_booking.cli train             # writes models/model.joblib + reports/metrics.json
PYTHONPATH=src python -m airbnb_first_booking.cli predict --input data/raw/test_users.csv
```

`make download-data` installs the `kaggle` CLI if missing, downloads the
competition archive into `data/raw/`, and extracts it — see
[`scripts/download_data.sh`](scripts/download_data.sh) and
[`data/raw/README.md`](data/raw/README.md) for authentication setup
(you'll need a `kaggle.json` API token).

`make` shortcuts for all of the above are available — see [`Makefile`](Makefile)
(`make install`, `make download-data`, `make profile`, `make train`, `make api`, `make docker-run`, `make test`).

### Notebooks (the analytical story)

The `notebooks/` folder mirrors this same pipeline as a guided narrative —
business framing, data understanding, EDA, feature engineering and modeling —
and is the best place to see *why* each design decision was made. See
[`notebooks/README.md`](notebooks/README.md) for the suggested reading order.
To run them locally: `pip install -r requirements.txt && jupyter lab`.

More deployment detail (architecture, configuration, request/response shape)
lives in [`docs/deployment.md`](docs/deployment.md).

## 4. Solution Strategy

1. **Data Description:** validate schema, dimensions, missing values, types and granularity.
2. **Feature Engineering:** create domain-driven variables (signup dates, account age, channel/device signals).
3. **Data Filtering:** remove records with no analytical value or leakage risk (e.g. `date_first_booking`, only known after the outcome).
4. **Exploratory Data Analysis:** validate hypotheses and separate relevant signal from noise.
5. **Data Preparation:** impute, scale and one-hot encode variables inside a single scikit-learn `Pipeline`.
6. **Feature Selection:** separate IDs, target, input variables and dropped columns via `configs/project.toml`.
7. **Machine Learning Modelling:** train a reproducible baseline (logistic regression) and an ensemble alternative (`HistGradientBoostingClassifier`), selectable through configuration.
8. **Evaluation:** compute technical metrics (accuracy, balanced accuracy, macro F1/precision/recall) *and* the ranking metric that actually matters for the business action — NDCG@5.
9. **Business Translation:** turn metrics into a decision framework for personalized communication and campaign targeting.
10. **Delivery:** persist reports, the trained pipeline and a predictions file, and serve everything through a versioned, containerized API.

## 5. Data Source

Source: Kaggle [Airbnb Recruiting New User Bookings](https://www.kaggle.com/competitions/airbnb-recruiting-new-user-bookings).

**Raw data is intentionally not committed to this repository** — `sessions.csv`
alone is ~600 MB, well past GitHub's 100 MB limit, and shipping large binary
data in a portfolio repo is poor practice anyway. Instead, `data/raw/` is
populated on demand straight from Kaggle:

```bash
make download-data
```

(See [`scripts/download_data.sh`](scripts/download_data.sh) and
[`data/raw/README.md`](data/raw/README.md) for the one-time Kaggle API token setup.)

Expected files once downloaded:

- `train_users_2.csv`
- `test_users.csv`
- `sessions.csv`
- `countries.csv`
- `age_gender_bkts.csv`
- `sample_submission_NDF.csv`

The pipeline combines user signup data, session behavior and auxiliary reference tables.

## 6. Development Journey

The notebooks are organized to show the evolution of the analysis, from
problem framing to the business translation of the results. They are also the
best place to grab portfolio screenshots:

- [`notebooks/00_business_understanding.ipynb`](notebooks/00_business_understanding.ipynb)
- [`notebooks/01_data_understanding.ipynb`](notebooks/01_data_understanding.ipynb)
- [`notebooks/02_exploratory_analysis.ipynb`](notebooks/02_exploratory_analysis.ipynb)
- [`notebooks/03_feature_engineering.ipynb`](notebooks/03_feature_engineering.ipynb)
- [`notebooks/04_modeling_and_business_results.ipynb`](notebooks/04_modeling_and_business_results.ipynb)

## 7. Top Data Insights and Hypotheses

- Users with higher session activity tend to leave the `NDF` ("no booking") class.
- Affiliate channel and initial device hint at different destination profiles.
- Implausible age values (e.g. birth years stored as ages) need correction so they don't contaminate the model.

## 8. Model and Evaluation Approach

A single scikit-learn `Pipeline` couples preprocessing (median/most-frequent
imputation, scaling and one-hot encoding via a `ColumnTransformer`) with the
estimator, so the exact same artifact can be trained, evaluated, batch-scored
and served — avoiding train/serve skew. Two estimators are available through
`configs/project.toml` (`[modeling].model_type`):

- `logistic_regression` (default): fast, interpretable linear baseline with `class_weight = "balanced"`.
- `gradient_boosting`: `HistGradientBoostingClassifier`, a histogram-based ensemble that handles large tabular datasets and mixed feature types well — the natural next step after a linear baseline, as recommended throughout *Hands-On Machine Learning*.

Evaluation reports both technical classification metrics and **NDCG@5**, the
ranking-aware metric used by the Kaggle leaderboard — because for this
business action, getting the right destination into a top-5 list matters more
than a single point prediction.

## 9. Performance and Business Results

A reproduced data profile is available at [`reports/data_profile.json`](reports/data_profile.json): 213,451 rows and 16 columns analyzed.

Main pipeline outputs:

- [`reports/metrics.json`](reports/metrics.json) — accuracy, balanced accuracy, macro F1/precision/recall and `ndcg_at_5`.
- `models/model.joblib` — the trained, ready-to-serve pipeline.

## 10. Business Translation

The top-N destination scores can drive prioritized recommendations,
personalized marketing campaigns and a meaningful reduction in generic
communication sent to brand-new users — directly supporting the original
business goal of engaging users earlier and more relevantly in their journey.

## 11. Repository Structure

- [`configs/project.toml`](configs/project.toml): single source of truth for data paths, target, metrics and modeling parameters.
- [`src/airbnb_first_booking/`](src/airbnb_first_booking/): modular Python package for data loading, feature engineering, modeling, analysis and serving.
- [`notebooks/`](notebooks/): the analytical journey, told through notebooks that reuse the same package code.
- [`data/raw/`](data/raw/): empty in git — populated on demand via `make download-data` (see [`scripts/download_data.sh`](scripts/download_data.sh)).
- [`reports/`](reports/): metrics, profiles and other generated results.
- `models/`: the trained model artifact, ready to be mounted into the API container.
- [`Dockerfile`](Dockerfile) / [`docker-compose.yml`](docker-compose.yml): containerized, one-command deployment of the prediction API.

## 12. Running on Google Colab

1. Open a new notebook on Google Colab.
2. Generate a Kaggle API token at Kaggle > Account > API > Create New Token.
3. Run the cells below.

Clone the repository and install dependencies:

```python
REPO_URL = "https://github.com/<your-username>/<this-repository>.git"
!git clone {REPO_URL} project
%cd project
!python -m pip install -q -r requirements.txt
```

Download or prepare the data:

```python
from google.colab import files
files.upload()  # upload your kaggle.json file

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/kaggle.json
!chmod 600 ~/.kaggle/kaggle.json
!make download-data
```

Run the main pipeline:

```python
!PYTHONPATH=src python -m airbnb_first_booking.cli validate-config
!PYTHONPATH=src python -m airbnb_first_booking.cli profile
!PYTHONPATH=src python -m airbnb_first_booking.cli train
```

## 13. Tests

```bash
python -m pytest
# or: make test
```

## 14. Next Steps to Improve

- Run a systematic comparison between the linear baseline and the gradient
  boosting option (`model_type = "gradient_boosting"`) with cross-validated NDCG@5.
- Explore session-level aggregated features from `sessions.csv` (the richest, currently underused signal).
- Add a temporal validation split that better mirrors the production scenario (predicting *future* signups).
- Generate a Kaggle-formatted submission file with the top-5 destinations per user.
