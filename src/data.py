"""Download, load, clean, and save the Telco Customer Churn dataset."""
from pathlib import Path

import pandas as pd
import requests

from src.config import (
    DATA_URL,
    PROCESSED_DATA_PATH,
    RAW_DATA_PATH,
    TARGET_COLUMN,
)


def download_raw_data(force: bool = False) -> Path:
    """Download the raw CSV from DATA_URL if it isn't already on disk."""
    if RAW_DATA_PATH.exists() and not force:
        print(f"[data] Raw file already present: {RAW_DATA_PATH}")
        return RAW_DATA_PATH

    print(f"[data] Downloading from {DATA_URL} ...")
    response = requests.get(DATA_URL, timeout=30)
    response.raise_for_status()
    RAW_DATA_PATH.write_bytes(response.content)
    print(f"[data] Saved raw data to {RAW_DATA_PATH}")
    return RAW_DATA_PATH


def load_raw_data() -> pd.DataFrame:
    """Load the raw CSV, downloading it first if necessary."""
    if not RAW_DATA_PATH.exists():
        download_raw_data()
    return pd.read_csv(RAW_DATA_PATH)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop customerID, fix TotalCharges, drop blanks, encode target."""
    df = df.copy()
    df = df.drop(columns=["customerID"])
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    rows_before = len(df)
    df = df.dropna(subset=["TotalCharges"]).reset_index(drop=True)
    print(f"[data] Dropped {rows_before - len(df)} rows with missing TotalCharges.")

    df[TARGET_COLUMN] = df[TARGET_COLUMN].map({"Yes": 1, "No": 0}).astype(int)
    return df


def save_processed(df: pd.DataFrame) -> Path:
    """Write the cleaned dataframe to CSV."""
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"[data] Saved cleaned data to {PROCESSED_DATA_PATH}")
    return PROCESSED_DATA_PATH


def run_pipeline(force_download: bool = False) -> pd.DataFrame:
    """End-to-end: download -> load -> clean -> save."""
    download_raw_data(force=force_download)
    raw = load_raw_data()
    clean = clean_data(raw)
    save_processed(clean)
    return clean


if __name__ == "__main__":
    run_pipeline(force_download=True)