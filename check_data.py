"""Quick profile of the raw source file (columns, missing values, salary units).

Run from the project root:  python check_data.py
"""
from pathlib import Path

import pandas as pd

SOURCE = Path("data/tech_jobs_2026/jobs_large.parquet")

if not SOURCE.exists():
    print(f"Raw file not found: {SOURCE}")
else:
    df = pd.read_parquet(SOURCE)
    print("Total rows:", f"{len(df):,}")
    print("\nColumns:", df.columns.tolist())
    print("\nShare missing per column:")
    print(df.isna().mean().round(3).to_string())
    print("\nSalary currencies:")
    print(df["salary_currency"].value_counts(dropna=False).head(10).to_string())
    print("\nSalary periods:")
    print(df["salary_period"].value_counts(dropna=False).to_string())
    print("\nPosting date range:", df["posted_at"].min(), "to", df["posted_at"].max())
