#!/usr/bin/env bash
# Downloads the Airbnb Recruiting New User Bookings dataset from Kaggle into
# data/raw/, so the raw competition files never need to be committed to git.
#
# Requirements:
#   - A Kaggle account and API token (kaggle.json), see:
#     https://www.kaggle.com/docs/api#authentication
#   - Either ~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY env vars set.
#
# Usage:
#   bash scripts/download_data.sh
#   make download-data

set -euo pipefail

COMPETITION="airbnb-recruiting-new-user-bookings"
TARGET_DIR="data/raw"

if ! command -v kaggle >/dev/null 2>&1; then
    echo "Installing the Kaggle CLI..."
    python -m pip install --quiet kaggle
fi

mkdir -p "$TARGET_DIR"

echo "Downloading '$COMPETITION' into $TARGET_DIR ..."
kaggle competitions download -c "$COMPETITION" -p "$TARGET_DIR"

echo "Extracting archives..."
find "$TARGET_DIR" -maxdepth 1 -name "*.zip" -exec unzip -q -o {} -d "$TARGET_DIR" \;
find "$TARGET_DIR" -maxdepth 1 -name "*.zip" -delete

echo "Done. Files available in $TARGET_DIR:"
ls -lh "$TARGET_DIR"
