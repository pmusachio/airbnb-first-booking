# Raw Data

Source: Kaggle [Airbnb Recruiting New User Bookings](https://www.kaggle.com/competitions/airbnb-recruiting-new-user-bookings).

This folder is **intentionally empty in git** (see the repository root
`.gitignore`) — the raw competition files are large (`sessions.csv` alone is
about 600 MB, well over GitHub's 100 MB limit) and are downloaded on demand
instead of being versioned.

Expected files once downloaded:

- `train_users_2.csv`
- `test_users.csv`
- `sessions.csv`
- `countries.csv`
- `age_gender_bkts.csv`
- `sample_submission_NDF.csv`

The pipeline combines user signup data, session behavior and auxiliary
reference tables.

## Download automatically via the Kaggle API

1. Create a Kaggle API token at *Kaggle > Account > API > Create New Token*
   and place `kaggle.json` at `~/.kaggle/kaggle.json` (or export
   `KAGGLE_USERNAME` / `KAGGLE_KEY`). See the
   [Kaggle API docs](https://www.kaggle.com/docs/api#authentication).
2. Run:

```bash
make download-data
# equivalent to: bash scripts/download_data.sh
```

This installs the `kaggle` CLI if needed, downloads the competition archive
into `data/raw/`, extracts it, and removes the `.zip` files — leaving the
folder ready for `make profile` / `make train`.

## Manual download

```bash
mkdir -p data/raw
kaggle competitions download -c airbnb-recruiting-new-user-bookings -p data/raw
find data/raw -maxdepth 1 -name "*.zip" -exec unzip -q -o {} -d data/raw \;
```
