# Airbnb First Booking — Destination Ranker

> Multiclass classification · Ranking (NDCG@5) · Imbalanced 12-class problem

## Business Problem

When a user signs up for Airbnb, personalizing onboarding — recommended listings, content,
language, currency — depends on where they are likely to travel. The decision the model informs is
**how to tailor the first-session experience**, by ranking the destinations a new user is most
likely to book first (12 outcomes, including NDF: no booking in the window).

Most users do not book at all (NDF is ~58%), so single-label accuracy is a trap: always predicting
NDF already scores 58% while telling the product team nothing. The useful output is a **ranking** of
likely destinations, judged by NDCG@5 (the competition's metric) — does the true destination appear
high in the top-5? The cost of error is a missed personalization opportunity, not a hard
misclassification.

## Dataset

[Airbnb New User Bookings](https://www.kaggle.com/c/airbnb-recruiting-new-user-bookings) (consent-gated competition).

| Property | Value |
|----------|-------|
| Users | 213,451 |
| Target | `country_destination` (12 classes) |
| Majority class | NDF, 58.3% |
| Fields | demographics, signup method/app, affiliate, device, browser, dates |

The full `train_users_2.csv` trains the model; a real stratified 20k subsample is versioned as the
offline fallback. The competition is consent-gated, so reproducing the reported (full-data) metrics
requires placing `train_users_2.csv` in `data/raw/`.

## Solution Strategy

1. **Acquisition** — use the real competition file in `data/raw` if present, otherwise the versioned subsample.
2. **Leakage control** — `date_first_booking` is dropped: it is populated only for users who booked, so it is null exactly when the label is NDF and would leak the target.
3. **Feature engineering** — account-creation calendar parts, days between first activity and signup, and age cleaning (out-of-range and birth-year entries), inside the model `Pipeline`.
4. **Imbalance** — because the goal is ranking under a dominant NDF class, the model is selected on log-loss (a proper probability score) and judged on NDCG@5; class re-weighting was tried but degraded ranking calibration, so it is not used.
5. **Model selection** — `StratifiedKFold` cross-validation compares a multinomial logistic model and histogram gradient boosting on log-loss; the winner is tuned with `RandomizedSearchCV`.
6. **Evaluation** — accuracy, top-5 accuracy, NDCG@5 and macro-F1 on a stratified holdout, against a majority-class baseline.

## Top Insights & Hypotheses

- **Age is the strongest predictor** (permutation importance 0.13), followed by account-creation timing — when and how a user signs up carries most of the destination signal.
- **The model beats the trivial baseline on accuracy and adds ranking value** the baseline cannot: it places the true destination in its top 5 for **96% of users** (NDCG@5 0.82).
- **Signup method and device** further separate bookers from NDF and steer destination probabilities.
- **Macro-F1 stays low** because the minority destinations are rarely the single most-likely outcome — expected when NDF dominates, and the reason ranking, not single-label accuracy, is the right objective.

## Engineered Features

| Feature | Definition | Business signal |
|---------|-----------|-----------------|
| account_year / month / dayofweek | calendar parts of signup date | seasonality and cohort effects |
| days_active_before_signup | days between first activity and account creation | consideration time before committing |
| age (cleaned) | out-of-range and birth-year entries corrected | reliable demographic input |

## Model

A histogram gradient boosting classifier (selected by cross-validation on log-loss, tuned with
randomized search) inside a `Pipeline` that owns the engineering and encoding. The majority-class
predictor sets the accuracy floor.

| Model | Holdout accuracy | Top-5 accuracy | NDCG@5 | Macro-F1 |
|-------|-----------------:|---------------:|-------:|---------:|
| Always-NDF baseline | 0.583 | — | — | — |
| **Hist gradient boosting (final)** | **0.630** | **0.959** | **0.824** | 0.102 |

## Business Results

The model lifts accuracy to **63%** (versus the 58% always-NDF floor) and, more importantly,
delivers a **ranked top-5 of destinations that contains the true outcome for 96% of users
(NDCG@5 0.82)**. Onboarding can use this ranking to pre-load relevant destinations, content and
language for each new user, focusing effort on the minority who will actually book.

## How to Run

1. **Clone**
   ```
   git clone https://github.com/pmusachio/airbnb-first-booking.git
   cd airbnb-first-booking
   ```
2. **Environment**
   ```
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. **Data** — a real 20k subsample is versioned under `data/sample/`. For the full reported metrics, accept the competition rules on Kaggle and place `train_users_2.csv` in `data/raw/`.
4. **Run the pipeline**
   ```
   python -m src.pipeline
   ```
5. **Tests**
   ```
   pytest tests/
   ```
6. **App (local)**
   ```
   streamlit run app/streamlit_app.py
   ```
7. **Live app** — [airbnb-first-booking.onrender.com](https://airbnb-first-booking.onrender.com) — rank a new user's likely destinations.

## Next Steps

- Add the `sessions.csv` behavioural features (action counts, time per action), the strongest predictors in the competition, to lift NDCG@5 further.
- Optimize NDCG@5 directly with a learning-to-rank objective rather than a proxy log-loss, and calibrate the booking-vs-NDF split as a first-stage gate.
- Model destination choice conditional on having booked, since the NDF-vs-booked decision and the where-to-book decision have different drivers.
