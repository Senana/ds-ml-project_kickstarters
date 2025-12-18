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


def identify_trending_categories(
    df: pd.DataFrame,
    lookback_weeks: int = 4,
    success_rate_threshold: Optional[float] = None
) -> pd.Series:
    """
    Identify trending categories by week using historical data.
    
    A category is considered "trending" if it has a high success rate in recent weeks.
    This is simpler than momentum-based approaches and captures categories that are
    performing well recently.
    """
    # Check required columns
    required_cols = ['launched', 'main_category', 'target']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Ensure launched is datetime
    df = df.copy()
    df['launched'] = pd.to_datetime(df['launched'], errors='coerce')
    
    # Create week identifier (year-week)
    df['launch_year_week'] = df['launched'].dt.to_period('W')
    
    # Sort by launch date for time-safe processing
    df_sorted = df.sort_values('launched').copy()
    
    # Initialize result series
    is_trending = pd.Series(False, index=df_sorted.index)
    
    # Get unique weeks
    unique_weeks = sorted(df_sorted['launch_year_week'].dropna().unique())
    
    # Process each week
    for i, current_week in enumerate(unique_weeks):
        # Get projects launched in this week
        current_week_mask = df_sorted['launch_year_week'] == current_week
        
        if current_week_mask.sum() == 0:
            continue
        
        # Get historical data: projects launched in the past N weeks (excluding current week)
        # Convert period to datetime for comparison
        current_week_start = current_week.to_timestamp()
        lookback_start = current_week_start - pd.Timedelta(weeks=lookback_weeks)
        
        historical_mask = (
            (df_sorted['launched'] >= lookback_start) &
            (df_sorted['launched'] < current_week_start)
        )
        
        if historical_mask.sum() == 0:
            # No historical data, skip this week
            continue
        
        historical_df = df_sorted[historical_mask]
        
        # Calculate category success rates from historical data
        category_stats = historical_df.groupby('main_category').agg({
            'target': ['count', 'mean']  # count = volume, mean = success rate
        }).reset_index()
        category_stats.columns = ['main_category', 'volume', 'success_rate']
        
        # Filter categories with minimum volume (at least 5 projects)
        category_stats = category_stats[category_stats['volume'] >= 5]
        
        if category_stats.empty:
            continue
        
        # Determine threshold: use 75th percentile if not specified
        if success_rate_threshold is None:
            threshold = category_stats['success_rate'].quantile(0.75)
        else:
            threshold = success_rate_threshold
        
        # Categories with high success rate are trending
        trending_categories = set(
            category_stats[category_stats['success_rate'] >= threshold]['main_category'].tolist()
        )
        
        # Mark projects in current week if their category is trending
        current_week_categories = df_sorted.loc[current_week_mask, 'main_category']
        is_trending.loc[current_week_mask] = current_week_categories.isin(trending_categories)
    
    # Return series aligned with original dataframe index
    return is_trending.reindex(df.index, fill_value=False)


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
        bins=[0, 15, 29, 45, 60, 75, float("inf")],
        labels=["<2 weeks", "2 weeks", "4 weeks", "6 weeks", "8 weeks", "8+ weeks"],
        right=True,
        include_lowest=True
    )

    # --- backers ---
    df["backers_per_pledged"] = df["backers"] / df["usd_pledged_real"]
    df["backer_pledged_bins"] = pd.qcut(
        df["backers_per_pledged"], q=5, labels=labels
    )

    # --- seasons ---
    df['launch_season'] = df['launched_month'].apply(convert_season)
    df['deadline_season'] = df['deadline_month'].apply(convert_season)

    # --- trending categories ---
    # Identify categories with high recent success rates
    # Simple approach: categories in top 25% of success rates in recent weeks
    logger.info("Computing trending category feature...")
    df['is_trending_category'] = identify_trending_categories(
        df=df,
        lookback_weeks=4,  # Look back 4 weeks
        success_rate_threshold=None  # Use 75th percentile of category success rates
    )
    logger.info(f"Trending category feature computed: {df['is_trending_category'].sum():,} projects ({df['is_trending_category'].mean():.2%}) belong to trending categories")

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