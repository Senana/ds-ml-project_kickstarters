import pandas as pd
import numpy as np
from pathlib import Path

import logging
from typing import List, Optional

from utils.clean_df import clean_kickstarter_data


def convert_season(month: Optional[int]) -> Optional[str]:
    """Convert month to season."""
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    elif month in [9, 10, 11]:
        return 'Fall'
    else:
        return None


def build_features(
    input_path: Path,
    output_path: Path,
    raw_path: Path,
    logger: logging.Logger,
) -> None:
    
# --- Dependency Check ---
    if not input_path.exists():
        logger.warning(f"Input file not found: {input_path}")
        logger.info(f"Triggering cleaning pipeline using raw data: {raw_path}")
        
        if not raw_path.exists():
            raise FileNotFoundError(f"Neither the input file '{input_path}' nor the raw file '{raw_path}' could be found.")
            
        # Trigger the cleaning function
        # Note: clean_kickstarter_data saves to output_dir, so we pass input_path.parent
        clean_kickstarter_data(
            raw_path=raw_path,
            output_dir=input_path.parent,
            logger=logger
        )
        logger.info("Cleaning finished. Resuming feature engineering...")

    logger.info("Starting feature engineering pipeline")

    df = pd.read_csv(input_path, low_memory=False, encoding="latin-1")
    logger.info(f"Loaded {len(df)} rows")

    # --- cleaning ---
    df = df[
        (df["usd_goal_real"] > 0) &
        (df["usd_pledged_real"] > 0)
    ].copy()

    # --- categories ---
    category_map = {
        'Art': 'Creative', 'Comics': 'Creative', 'Crafts': 'Creative',
        'Dance': 'Creative', 'Design': 'Creative',
        'Fashion': 'Consumer', 'Food': 'Consumer',
        'Film & Video': 'Entertainment', 'Games': 'Entertainment',
        'Music': 'Entertainment', 'Theater': 'Entertainment',
        'Photography': 'Creative', 'Publishing': 'Creative',
        'Technology': 'Tech', 'Journalism': 'Other'
    }
    df["main_category_grouped"] = (
        df["main_category"].map(category_map).fillna("Other")
    )

    # --- continents ---
    continent_map = {
        'US': 'North America', 'CA': 'North America', 'MX': 'North America',
        'GB': 'Europe', 'DE': 'Europe', 'FR': 'Europe', 'IT': 'Europe',
        'ES': 'Europe', 'NL': 'Europe', 'IE': 'Europe', 'SE': 'Europe',
        'CH': 'Europe', 'AT': 'Europe', 'DK': 'Europe', 'BE': 'Europe',
        'LU': 'Europe', 'NO': 'Europe',
        'AU': 'Oceania', 'NZ': 'Oceania',
        'JP': 'Asia', 'SG': 'Asia', 'HK': 'Asia'
    }

    df["continent"] = df["country"].map(continent_map).fillna("Other")

    # --- time ---
    df["launched"] = pd.to_datetime(df["launched"], errors="coerce")
    df["deadline"] = pd.to_datetime(df["deadline"], errors="coerce")

    for col in ["launched", "deadline"]:
        df[f"{col}_year"] = df[col].dt.year
        df[f"{col}_month"] = df[col].dt.month

    # --- financial bins ---
    labels = ["Very Low", "Low", "Medium", "High", "Very High"]
    df["usd_goal_bins"] = pd.qcut(df["usd_goal_real"], q=5, labels=labels)
    df["usd_pledged_bins"] = pd.qcut(df["usd_pledged_real"], q=5, labels=labels)

    # --- category averages ---
    df['pledged_per_category'] = df.groupby('main_category')['usd_pledged_real'].transform('mean')
    df['goal_per_category'] = df.groupby('main_category')['usd_goal_real'].transform('mean')

    # --- category percentiles ---
    df["category_goal_percentile"] = (
        df.groupby("main_category_grouped")["usd_goal_real"]
        .transform(lambda x: pd.qcut(x, q=5, labels=labels, duplicates="drop"))
    )

    # --- duration ---
    df["duration_bins"] = pd.cut(
        df["duration_days"],
        bins=[15, 29, 45, 60, 75],
        labels=["2 weeks", "4 weeks", "6 weeks", "8 weeks"]
    )

    # --- backers ---
    df["backers_per_pledged"] = df["backers"] / df["usd_pledged_real"]
    df["backer_pledged_bins"] = pd.qcut(
        df["backers_per_pledged"], q=5, labels=labels
    )

    # --- seasons ---
    df['launch_season'] = df['launched_month'].apply(convert_season)
    df['deadline_season'] = df['deadline_month'].apply(convert_season)

    # --- save ---
    logger.info(f"Final columns before save: {df.columns.tolist()}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    logger.info(f"Saved engineered dataset to {output_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    BASE_DIR = Path.cwd()

    build_features(
        input_path=BASE_DIR / "data/cleaned/kickstarter_cleaned.csv",
        output_path=BASE_DIR / "data/feature/kickstarter_featured.csv",
        raw_path=BASE_DIR / "data/raw/ks-projects-201801.csv",
        logger=logger
    )