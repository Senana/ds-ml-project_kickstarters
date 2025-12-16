import pandas as pd
import numpy as np
from pathlib import Path

import logging
from typing import List

def clean_kickstarter_data(
    raw_path: Path,
    output_dir: Path,
    logger: logging.Logger,
    main_states: List[str] = ("successful", "failed"),
    cancelled_state: str = "canceled",
) -> None:
    logger.info("Starting Kickstarter data cleaning")

    # --- load ---
    df = pd.read_csv(raw_path)
    logger.info(f"Loaded {len(df):,} rows × {len(df.columns)} columns")

    # --- drop columns ---
    cols_to_drop = [
        "name", "category", "goal", "pledged",
        "currency", "usd pledged"
    ]
    existing = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=existing)
    logger.info(f"Dropped columns: {existing}")

    # --- datetime & duration ---
    df["launched"] = pd.to_datetime(df["launched"], errors="coerce")
    df["deadline"] = pd.to_datetime(df["deadline"], errors="coerce")
    df["duration_days"] = (df["deadline"] - df["launched"]).dt.days

    # --- normalize column names ---
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(" ", "_")
        .str.lower()
    )

    # --- fix country codes ---
    df["country"] = df["country"].replace('N,0"', 'NO')

    # --- state filtering ---
    df_main = df[df["state"].isin(main_states)].copy()
    df_main["target"] = (df_main["state"] == "successful").astype(int)

    df_with_cancelled = df[
        df["state"].isin(list(main_states) + [cancelled_state])
    ].copy()
    df_with_cancelled["target"] = (
        df_with_cancelled["state"] == "successful"
    ).astype(int)

    # --- drop state columns ---
    df_main = df_main.drop(columns=['state'])

    # --- save ---
    output_dir.mkdir(parents=True, exist_ok=True)

    main_path = output_dir / "kickstarter_cleaned.csv"
    cancelled_path = output_dir / "kickstarter_cleaned_with_cancelled.csv"

    df_main.to_csv(main_path, index=False)
    df_with_cancelled.to_csv(cancelled_path, index=False)

    logger.info("Data cleaning completed")
    logger.info(f"Saved: {main_path}")
    logger.info(f"Saved: {cancelled_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    BASE_DIR = Path.cwd()

    clean_kickstarter_data(
        raw_path=BASE_DIR / "data/raw/ks-projects-201801.csv",
        output_dir=BASE_DIR / "data/cleaned",
        logger=logger
    )