"""Step 1 of 3: clean the raw job-postings dump into one compact Parquet file.

Input : data/tech_jobs_2026/jobs_large.parquet  (raw scrape, not committed)
Output: data/jobs_clean.parquet                 (committed; read by the dashboard)
        data/data_summary.json                  (cleaning funnel shown in the dashboard)

Every posting is kept for counting purposes. Salary columns are only filled
when the posting advertises a plausible annual USD range (see features.py),
so salary statistics never mix currencies, hourly rates, or typos.

Run from the project root:  python pipeline/collector.py
"""
import json
from pathlib import Path

import pandas as pd

from features import clean_annual_usd, role_family, seniority

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "tech_jobs_2026" / "jobs_large.parquet"
OUTPUT = ROOT / "data" / "jobs_clean.parquet"
SUMMARY = ROOT / "data" / "data_summary.json"


def text_or_none(series: pd.Series) -> pd.Series:
    """Strip whitespace and turn empty strings into missing values."""
    s = series.astype("string").str.strip()
    return s.mask(s.eq("") | s.str.lower().isin(["nan", "none", "null"]))


def build_location(city: pd.Series, country: pd.Series, remote: pd.Series) -> pd.Series:
    """Readable location label: 'City, Country', 'Country', or 'Remote'."""
    label = (city + ", " + country).fillna(country)
    return label.fillna(remote.map({True: "Remote"})).fillna("Unknown")


def run():
    if not SOURCE.exists():
        raise FileNotFoundError(
            f"Missing raw dataset: {SOURCE}\n"
            "Download it from the source listed in README.md first.")

    raw = pd.read_parquet(SOURCE)
    n_raw = len(raw)

    # Every URL in the source is unique, but guard against a re-scrape anyway.
    raw = raw.drop_duplicates("url")

    df = pd.DataFrame({
        "title": text_or_none(raw["title"]),
        "company": text_or_none(raw["company_name"]),
        "industry": text_or_none(raw["company_industry"]),
        "department": text_or_none(raw["department"]),
        "country": text_or_none(raw["country"]),
        "city": text_or_none(raw["city"]),
        "remote": raw["is_remote"].astype(bool),
        "employment_type": text_or_none(raw["employment_type"]).str.replace("_", " ").str.title(),
        "posted": pd.to_datetime(raw["posted_at"], utc=True, format="mixed").dt.tz_localize(None).dt.normalize(),
        "skills": raw["skills"].map(lambda v: sorted({str(x).strip() for x in v if str(x).strip()})
                                    if v is not None else []),
        "url": raw["url"],
    })
    df["location"] = build_location(df["city"], df["country"], df["remote"])

    # Derived labels (shared with the model through features.py).
    df["role_family"] = df["title"].map(role_family)
    df["seniority"] = df["title"].map(seniority)

    # Salary: keep the advertised range and its midpoint only when plausible.
    mid, status = clean_annual_usd(raw["salary_min"], raw["salary_max"],
                                   raw["salary_currency"], raw["salary_period"])
    usable = status.eq("usable")
    df["salary_min"] = raw["salary_min"].where(usable)
    df["salary_max"] = raw["salary_max"].where(usable)
    df["salary_mid"] = mid
    df["salary_status"] = status

    # Low-cardinality text columns are stored as categories to keep the file small.
    for col in ["industry", "country", "employment_type", "role_family", "seniority", "salary_status"]:
        df[col] = df[col].astype("category")

    df.reset_index(drop=True).to_parquet(OUTPUT, index=False, compression="zstd")

    # Funnel numbers for the dashboard's Data & Methods tab.
    summary = {
        "source": "zalizedata/tech-job-postings-salary-dataset (CC BY-NC 4.0)",
        "raw_postings": int(n_raw),
        "postings": int(len(df)),
        "companies": int(df["company"].nunique()),
        "countries": int(df["country"].nunique()),
        "posted_min": str(df["posted"].min().date()),
        "posted_max": str(df["posted"].max().date()),
        "salary_status_counts": {k: int(v) for k, v in status.value_counts().items()},
        "usable_salaries": int(usable.sum()),
        "rules": {
            "currency_period": "USD per year only",
            "plausible_midpoint_usd": [20_000, 1_000_000],
            "max_range_ratio": 4.0,
        },
    }
    SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Raw postings:       {n_raw:,}")
    print(f"Cleaned postings:   {len(df):,}")
    print(f"Usable salaries:    {int(usable.sum()):,}")
    for label, count in status.value_counts().items():
        print(f"  {label:<20}{count:>9,}")
    print(f"Wrote {OUTPUT.relative_to(ROOT)} ({OUTPUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    run()
