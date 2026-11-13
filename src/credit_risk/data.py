"""Download, clean and (for tests) synthesise the credit default dataset.

Source: "Default of Credit Card Clients" (Yeh & Lien, 2009), UCI Machine Learning
Repository, id 350, licensed CC BY 4.0. 30,000 credit card clients in Taiwan (2005).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from credit_risk.config import (
    BILL_COLS,
    DATA_DIR,
    PAY_AMT_COLS,
    PAY_STATUS_COLS,
    RANDOM_STATE,
    RAW_COLUMNS,
    RAW_DATA_PATH,
    TARGET,
)

UCI_DATASET_ID = 350
UCI_PAGE = "https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients"


def _use_system_certificates() -> None:
    """Trust the operating system's certificate store.

    Corporate networks often inspect HTTPS traffic with their own certificate, which the OS
    trusts but Python's bundled list does not. `truststore` makes Python use the OS store,
    so verification stays on and works behind such networks.
    """
    try:
        import truststore

        truststore.inject_into_ssl()
    except ImportError:
        pass


def download() -> pd.DataFrame:
    """Fetch the dataset from UCI and give the columns readable names."""
    _use_system_certificates()
    from ucimlrepo import fetch_ucirepo

    try:
        ds = fetch_ucirepo(id=UCI_DATASET_ID)
    except ConnectionError as err:
        raise SystemExit(
            "\nCould not download the dataset automatically (network blocked?).\n"
            f"1. Open {UCI_PAGE} in your browser and click 'Download'.\n"
            "2. Unzip it and put 'default of credit card clients.xls' in the data/ folder.\n"
            '3. Run: python -m credit_risk.data --file "data/default of credit card clients.xls"\n'
        ) from err
    features = ds.data.features.copy()
    # UCI publishes the columns as X1..X23 in a fixed documented order.
    features.columns = RAW_COLUMNS
    df = features.assign(**{TARGET: ds.data.targets.iloc[:, 0].to_numpy()})
    return df


def read_local_file(path: Path) -> pd.DataFrame:
    """Read a manually downloaded copy (UCI .xls, or a .csv such as the Kaggle version).

    The column order is the same in every published copy (ID, then the 23 features, then
    the target), but the names differ (X1..X23, PAY_0 vs PAY_1, ...), so columns are
    renamed by position.
    """
    path = Path(path)
    reader = pd.read_csv if path.suffix.lower() == ".csv" else pd.read_excel
    df = reader(path)
    if "LIMIT_BAL" not in df.columns:  # UCI .xls has an extra X1..X23 header row on top
        df = reader(path, header=1)
    df = df.drop(columns=[c for c in df.columns if str(c).strip().upper() == "ID"])
    if df.shape[1] != len(RAW_COLUMNS) + 1:
        raise SystemExit(f"Expected {len(RAW_COLUMNS) + 1} columns in {path}, found {df.shape[1]}.")
    df.columns = [*RAW_COLUMNS, TARGET]
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fix undocumented category codes and enforce types."""
    df = df.copy()
    # EDUCATION: 1 grad school, 2 university, 3 high school, 4 other. 0/5/6 are undocumented.
    df["EDUCATION"] = df["EDUCATION"].where(df["EDUCATION"].isin([1, 2, 3, 4]), 4)
    # MARRIAGE: 1 married, 2 single, 3 other. 0 is undocumented.
    df["MARRIAGE"] = df["MARRIAGE"].where(df["MARRIAGE"].isin([1, 2, 3]), 3)
    df = df.astype({c: "int64" for c in [*PAY_STATUS_COLS, "SEX", "EDUCATION", "MARRIAGE", "AGE"]})
    df[TARGET] = df[TARGET].astype(int)
    return df.drop_duplicates().reset_index(drop=True)


def make_synthetic(n: int = 3000, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Generate data with the same schema, used for tests and CI (no download needed)."""
    rng = np.random.default_rng(seed)
    risk = rng.normal(size=n)  # latent creditworthiness
    limit = rng.choice([20_000, 50_000, 100_000, 200_000, 500_000], size=n).astype(float)
    df = pd.DataFrame({
        "LIMIT_BAL": limit,
        "SEX": rng.integers(1, 3, n),
        "EDUCATION": rng.integers(1, 5, n),
        "MARRIAGE": rng.integers(1, 4, n),
        "AGE": rng.integers(21, 70, n),
    })
    for i, col in enumerate(PAY_STATUS_COLS):
        delay = np.round(risk * 1.2 + rng.normal(scale=0.8, size=n) - 0.1 * i)
        df[col] = np.clip(delay, -2, 8).astype(int)
    for col in BILL_COLS:
        df[col] = (limit * np.clip(0.4 + 0.2 * risk + rng.normal(scale=0.2, size=n), 0, 1.2)).round()
    for b, p in zip(BILL_COLS, PAY_AMT_COLS, strict=True):
        df[p] = (df[b] * np.clip(0.3 - 0.15 * risk + rng.normal(scale=0.1, size=n), 0, 1)).round()
    logit = -1.4 + 1.3 * risk + rng.normal(scale=0.5, size=n)
    df[TARGET] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return df


def load(path=RAW_DATA_PATH) -> pd.DataFrame:
    return clean(pd.read_csv(path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the credit default dataset.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--synthetic", action="store_true", help="write synthetic data instead")
    source.add_argument("--file", type=Path, help="use a manually downloaded .xls/.xlsx/.csv file")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if args.synthetic:
        df = make_synthetic()
    elif args.file:
        df = read_local_file(args.file)
    else:
        df = download()
    df.to_csv(RAW_DATA_PATH, index=False)
    print(f"Saved {len(df):,} rows to {RAW_DATA_PATH} (default rate {df[TARGET].mean():.1%})")


if __name__ == "__main__":
    main()
