import pandas as pd
import numpy as np
from pathlib import Path

# Paths
BASE_DIR = Path.cwd().resolve().parents[1]
RAW_PATH = BASE_DIR / "data" / "raw"
CLEANED_PATH = BASE_DIR / "data" / "cleaned"

RAW_DATA_PATH = Path(RAW_PATH)
CLEANED_DATA_PATH = Path(CLEANED_PATH)

# Create output directory if not exists
CLEANED_DATA_PATH.mkdir(parents=True, exist_ok=True)

print("Setup complete")

# Load df2 (the larger/newer dataset)
# Adjust filename as needed
df = pd.read_csv(RAW_DATA_PATH / 'ks-projects-201801.csv')

print(f"Loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
print(f"\nColumns: {df.columns.tolist()}")

# Quick overview
df.info()

# Check state distribution before cleaning
print("State distribution (before):")
print(df['state'].value_counts())
print(f"\nTotal: {len(df):,}")

# Columns to drop
COLS_TO_DROP = [
    'name',
    'category',
    'goal',
    'pledged',
    'currency',
    'usd pledged',
]

# Check which columns exist before dropping
existing_to_drop = [col for col in COLS_TO_DROP if col in df.columns]
missing = [col for col in COLS_TO_DROP if col not in df.columns]

print(f"Will drop: {existing_to_drop}")
if missing:
    print(f"Not found (skipping): {missing}")


# Drop columns
df_clean = df.drop(columns=existing_to_drop)

print(f"Columns before: {len(df.columns)}")
print(f"Columns after: {len(df_clean.columns)}")
print(f"\nRemaining columns: {df_clean.columns.tolist()}")

# Transform dates into datetime
# deadline, launched - should be datetime types
df_clean["launched"] = pd.to_datetime(df_clean["launched"], errors="coerce")
df_clean["deadline"] = pd.to_datetime(df_clean["deadline"], errors="coerce")

# create new column - duration_days
df_clean["duration_days"] = (
    df_clean["deadline"] - df_clean["launched"]
).dt.days

# Columns clear naming
df_clean.columns = df_clean.columns.str.strip().str.replace(' ', '_').str.lower()
print(f"\nRenamed columns: {df_clean.columns.tolist()}")

# Define states
MAIN_STATES = ['successful', 'failed']
OPTIONAL_STATE = 'canceled'

# Check current distribution
print("Current state distribution:")
print(df_clean['state'].value_counts())

# Dataset 1: Only successful & failed
df_main = df_clean[df_clean['state'].isin(MAIN_STATES)].copy()

print(f"Main dataset (successful + failed):")
print(f"  Rows: {len(df_main):,}")
print(f"  Distribution:")
print(df_main['state'].value_counts())

# Dataset 2: Including cancelled
states_with_cancelled = MAIN_STATES + [OPTIONAL_STATE]
df_with_cancelled = df_clean[df_clean['state'].isin(states_with_cancelled)].copy()

print(f"Dataset with cancelled:")
print(f"  Rows: {len(df_with_cancelled):,}")
print(f"  Distribution:")
print(df_with_cancelled['state'].value_counts())

# For main dataset: successful=1, failed=0
df_main['target'] = (df_main['state'] == 'successful').astype(int)

print("Main dataset target distribution:")
print(df_main['target'].value_counts())

# For dataset with cancelled: successful=1, failed/canceled=0
df_with_cancelled['target'] = (df_with_cancelled['state'] == 'successful').astype(int)

print("Dataset with cancelled - target distribution:")
print(df_with_cancelled['target'].value_counts())

# Save main dataset
main_path = CLEANED_DATA_PATH / 'kickstarter_cleaned.csv'
df_main.to_csv(main_path, index=False)
print(f" Saved: {main_path}")

# Save dataset with cancelled
cancelled_path = CLEANED_DATA_PATH / 'kickstarter_cleaned_with_cancelled.csv'
df_with_cancelled.to_csv(cancelled_path, index=False)
print(f"\n Saved: {cancelled_path}")

print("="*60)
print("DATA CLEANING COMPLETE")
print("="*60)
print(f"""
Original dataset:
  - Rows: {len(df):,}
  - Columns: {len(df.columns)}

Dropped columns:
  - {existing_to_drop}

Main dataset (kickstarter_cleaned.csv):
  - States: successful, failed
  - Rows: {len(df_main):,}

With cancelled (kickstarter_cleaned_with_cancelled.csv):
  - States: successful, failed, canceled
  - Rows: {len(df_with_cancelled):,}

Output location: {CLEANED_DATA_PATH.absolute()}
""")

def clean_ks_data(input_path: Path, file_name: str, output_path: Path, canceled = False, main_states = ['successful', 'failed']):
    """
        Clear input data and save cleaned datasets.
    """

    # Create output directory if not exists
    input_path.mkdir(parents=True, exist_ok=True)

    # load dataset
    df = pd.read_csv(input_path / 'ks-projects-201801.csv')

