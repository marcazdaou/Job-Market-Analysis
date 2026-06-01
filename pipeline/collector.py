"""
Job Market Analytics - Data Collector

This script is the first step in the pipeline. It loads salary data from the
Kaggle dataset and stores it in a local SQLite database. If the Kaggle CSV is
not present, it falls back to generating synthetic data so the app still runs.

To use real data:
  1. Go to: https://www.kaggle.com/datasets/ruchi798/data-science-job-salaries
  2. Download ds_salaries.csv
  3. Place it at: data/ds_salaries.csv
  4. Run: python pipeline/collector.py
"""

import sqlite3
import json
import random
import os
import pandas as pd
from datetime import datetime, timedelta

# ── File paths ────────────────────────────────────────────────────────────────
# os.path.dirname(__file__) gives the folder this script lives in (pipeline/).
# We go one level up ("../") to reach the project root, then into data/.
DB_PATH  = os.path.join(os.path.dirname(__file__), "../data/jobs.db")
CSV_PATH = os.path.join(os.path.dirname(__file__), "../data/ds_salaries.csv")

# ── Static data pools ─────────────────────────────────────────────────────────
# Used when assigning company names (Kaggle dataset does not include them).
COMPANIES = [
    "Google", "Amazon", "Meta", "Microsoft", "Apple", "Netflix", "Uber", "Airbnb",
    "Stripe", "Databricks", "Snowflake", "Palantir", "Spotify", "LinkedIn",
    "Salesforce", "Oracle", "IBM", "Adobe", "Twilio", "MongoDB", "Cloudflare",
    "DoorDash", "Lyft", "Robinhood", "Coinbase", "Block", "Figma", "Notion",
    "JPMorgan Chase", "Goldman Sachs", "Capital One", "American Express",
    "Johnson & Johnson", "Pfizer", "Walmart", "Target", "Nike", "Tesla",
]

# Maps Kaggle's 2-letter country codes to readable city names for the dashboard.
# US has multiple cities so we can randomly spread records across them.
# Countries not listed here will just show as their country code.
COUNTRY_TO_LOCATION = {
    "US": ["San Francisco, CA", "New York, NY", "Seattle, WA", "Austin, TX",
           "Boston, MA", "Chicago, IL", "Los Angeles, CA", "Denver, CO",
           "Atlanta, GA", "San Jose, CA", "Washington, DC", "Dallas, TX"],
    "GB": ["London, UK"],
    "CA": ["Toronto, Canada"],
    "DE": ["Berlin, Germany"],
    "IN": ["Bangalore, India"],
    "FR": ["Paris, France"],
    "AU": ["Sydney, Australia"],
    "ES": ["Madrid, Spain"],
    "NL": ["Amsterdam, Netherlands"],
    "PT": ["Lisbon, Portugal"],
    "BR": ["Sao Paulo, Brazil"],
    "SG": ["Singapore"],
    "JP": ["Tokyo, Japan"],
    "MX": ["Mexico City, Mexico"],
    "IT": ["Milan, Italy"],
    "SE": ["Stockholm, Sweden"],
}

# Skills assigned based on keywords found in the job title.
# The Kaggle dataset does not include skills, so we generate them here.
# Each key maps to a list of skills commonly required for that type of role.
TITLE_SKILLS = {
    "engineer":  ["Python", "SQL", "Spark", "Airflow", "AWS", "Docker", "Kafka", "dbt", "Terraform"],
    "scientist": ["Python", "SQL", "Scikit-learn", "TensorFlow", "Statistics", "Pandas", "Machine Learning"],
    "analyst":   ["SQL", "Python", "Tableau", "Power BI", "Statistics", "Looker", "Data Modeling"],
    "ml":        ["Python", "TensorFlow", "PyTorch", "Scikit-learn", "AWS", "Docker", "Machine Learning"],
    "architect": ["SQL", "Python", "AWS", "Azure", "GCP", "Spark", "Kafka", "Data Warehousing", "Terraform"],
    "bi":        ["SQL", "Power BI", "Tableau", "Looker", "Python", "Data Modeling", "ETL"],
    "analytics": ["SQL", "dbt", "Python", "Looker", "Snowflake", "BigQuery", "Data Modeling"],
    "research":  ["Python", "Statistics", "Machine Learning", "TensorFlow", "Pandas", "NumPy", "R"],
    "default":   ["Python", "SQL", "AWS", "Pandas", "Git", "Statistics"],
}

# Extra skills randomly added to each job to add variety.
EXTRA_SKILLS = [
    "Kubernetes", "Scala", "Java", "R", "Redshift", "BigQuery", "Databricks",
    "Hadoop", "PostgreSQL", "MySQL", "MongoDB", "Redis", "Elasticsearch",
    "Git", "Linux", "REST APIs", "Azure", "GCP", "ELT", "ETL", "NumPy",
]

# Kaggle uses coded experience levels. This maps them to realistic year ranges.
# EN = Entry, MI = Mid, SE = Senior, EX = Executive/Principal
EXP_LEVEL_MAP = {"EN": (0, 2), "MI": (2, 5), "SE": (5, 9), "EX": (8, 15)}

# Kaggle uses coded employment types. This maps them to readable strings.
EMP_TYPE_MAP  = {"FT": "Full-time", "PT": "Part-time", "CT": "Contract", "FL": "Freelance"}

# ── Synthetic fallback data ───────────────────────────────────────────────────
# These are only used when ds_salaries.csv is not present.
TITLES = [
    "Data Engineer", "Senior Data Engineer", "Data Analyst", "Senior Data Analyst",
    "Data Scientist", "Senior Data Scientist", "ML Engineer", "Machine Learning Engineer",
    "Analytics Engineer", "Business Intelligence Engineer", "Data Architect",
    "Big Data Engineer", "ETL Developer", "Data Platform Engineer",
    "Staff Data Engineer", "Principal Data Engineer", "Lead Data Analyst",
    "Applied Scientist", "Research Scientist", "AI Engineer",
]
LOCATIONS = [
    "San Francisco, CA", "New York, NY", "Seattle, WA", "Austin, TX",
    "Boston, MA", "Chicago, IL", "Los Angeles, CA", "Denver, CO",
    "Atlanta, GA", "Miami, FL", "Remote", "San Jose, CA", "Washington, DC",
    "Portland, OR", "San Diego, CA", "Dallas, TX", "Raleigh, NC",
]
# Salary ranges (min, max) by title keyword, based on US market benchmarks.
SALARY_MAP = {
    "Principal": (180_000, 280_000), "Staff":      (170_000, 260_000),
    "Senior":    (140_000, 220_000), "Lead":       (150_000, 230_000),
    "Scientist": (130_000, 210_000), "Architect":  (150_000, 240_000),
    "ML Engineer":(140_000, 220_000),"AI Engineer":(135_000, 215_000),
    "Engineer":  (110_000, 180_000), "Analyst":    ( 75_000, 140_000),
    "Developer": (100_000, 160_000),
}


# ── Helper functions ──────────────────────────────────────────────────────────

def _skills_for_title(title: str) -> list:
    """
    Looks up the job title in TITLE_SKILLS to find the relevant skill set,
    picks a random subset of those base skills, then adds a few extra skills
    from EXTRA_SKILLS to simulate the variety seen in real job postings.
    dict.fromkeys() removes duplicates while preserving order.
    """
    title_lower = title.lower()
    for keyword, base in TITLE_SKILLS.items():
        if keyword in title_lower:
            chosen = random.sample(base, min(len(base), random.randint(4, 7)))
            extras = random.sample(EXTRA_SKILLS, random.randint(1, 3))
            return list(dict.fromkeys(chosen + extras))
    # If no keyword matched, use the default skill set
    chosen = TITLE_SKILLS["default"].copy()
    extras = random.sample(EXTRA_SKILLS, random.randint(2, 4))
    return list(dict.fromkeys(chosen + extras))


def _location_for_country(country_code: str, is_remote: bool) -> str:
    """
    Converts a 2-letter country code from the Kaggle dataset into a readable
    city name. Remote jobs always return "Remote" regardless of country.
    If the country code is not in our mapping, we just use the code as-is.
    """
    if is_remote:
        return "Remote"
    options = COUNTRY_TO_LOCATION.get(country_code, [country_code])
    return random.choice(options)


# ── Database setup ────────────────────────────────────────────────────────────

def init_db():
    """
    Creates the SQLite database file and the jobs table.
    We DROP the table first so that re-running the collector always starts
    fresh — otherwise records would be duplicated on each run.
    """
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DROP TABLE IF EXISTS jobs")  # always start with a clean table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            title       TEXT,        -- job title (e.g. "Senior Data Engineer")
            company     TEXT,        -- company name (synthetic for all records)
            location    TEXT,        -- city or "Remote"
            salary_min  REAL,        -- estimated lower bound of salary range
            salary_max  REAL,        -- estimated upper bound of salary range
            salary_avg  REAL,        -- the main salary figure used everywhere
            skills      TEXT,        -- JSON array of skill strings
            experience  INTEGER,     -- years of experience
            job_type    TEXT,        -- "Full-time", "Contract", etc.
            remote      INTEGER,     -- 1 = remote, 0 = on-site
            date_posted TEXT,        -- date string "YYYY-MM-DD"
            source      TEXT         -- "kaggle_ds_salaries" or "synthetic"
        )
    """)
    conn.commit()
    conn.close()
    print("Database initialized.")


# ── Kaggle data loader ────────────────────────────────────────────────────────

def load_from_kaggle(csv_path: str) -> list[dict]:
    """
    Reads the Kaggle ds_salaries.csv file and transforms each row into a dict
    that matches the jobs table schema.

    Key decisions:
    - We only keep US companies because international salaries converted to USD
      vary wildly by country and would confuse the ML model (a Data Scientist
      in India earns ~$20k USD; the same role in the US earns ~$150k USD).
      Without a country feature, the model cannot learn this difference.
    - salary_min/salary_max are estimated as +/- 10% of the reported figure
      since the dataset only provides a single salary value.
    - Skills are synthetically generated because the Kaggle dataset does not
      include them. They are assigned based on the job title.
    - Company names are randomly assigned from COMPANIES for the same reason.
    - date_posted is randomly assigned within the past 60 days because the
      dataset records the year but not the posting date.
    """
    df = pd.read_csv(csv_path)

    # Keep only US-based companies for salary comparability
    df = df[df["company_location"] == "US"].copy()
    # Remove any entries below $50k (data entry errors or misreported currencies)
    df = df[df["salary_in_usd"] >= 50_000].copy()

    jobs = []
    today = datetime.now()

    for _, row in df.iterrows():
        title      = str(row.get("job_title", "Data Analyst")).strip()
        salary_usd = float(row.get("salary_in_usd", 0))

        # Convert coded experience level to a random integer year value
        exp_code   = str(row.get("experience_level", "MI"))
        lo, hi     = EXP_LEVEL_MAP.get(exp_code, (2, 5))
        experience = random.randint(lo, hi)

        # Convert coded employment type to a readable string
        emp_code   = str(row.get("employment_type", "FT"))
        job_type   = EMP_TYPE_MAP.get(emp_code, "Full-time")

        # The dataset has two formats: older uses remote_ratio (0/50/100),
        # newer uses work_setting ("Remote"/"In-person"/"Hybrid"). We handle both.
        remote_ratio = int(row["remote_ratio"]) if "remote_ratio" in df.columns else 0
        work_setting = str(row["work_setting"])  if "work_setting" in df.columns else ""
        is_remote    = (remote_ratio == 100) or (work_setting.lower() == "remote")

        country  = str(row.get("company_location", "US"))
        location = _location_for_country(country, is_remote)

        # Simulate a posting date within the last 60 days
        days_ago    = random.randint(0, 60)
        date_posted = (today - timedelta(days=days_ago)).strftime("%Y-%m-%d")

        jobs.append({
            "title":       title,
            "company":     random.choice(COMPANIES),
            "location":    location,
            "salary_min":  round(salary_usd * 0.9, -3),   # 10% below reported salary
            "salary_max":  round(salary_usd * 1.1, -3),   # 10% above reported salary
            "salary_avg":  salary_usd,                     # actual reported salary
            "skills":      json.dumps(_skills_for_title(title)),
            "experience":  experience,
            "job_type":    job_type,
            "remote":      int(is_remote),
            "date_posted": date_posted,
            "source":      "kaggle_ds_salaries",           # tag so we can tell real from synthetic
        })

    return jobs


# ── Synthetic data generator ──────────────────────────────────────────────────

def _synthetic_salary(title: str):
    """
    Looks up the title in SALARY_MAP to get a realistic salary range,
    then adds random variance (+/- $10k) so all records don't look identical.
    Rounds to the nearest $1,000 for realism.
    Returns (sal_min, sal_max, sal_avg).
    """
    for keyword, (lo, hi) in SALARY_MAP.items():
        if keyword in title:
            v = random.uniform(-10_000, 10_000)
            sal_min = round(lo + v, -3)
            sal_max = round(hi + v, -3)
            return sal_min, sal_max, (sal_min + sal_max) / 2
    # Default salary if no keyword matched
    return 90_000, 150_000, 120_000


def generate_synthetic(n: int = 2000) -> list[dict]:
    """
    Generates n fully synthetic job postings. Used in two cases:
    1. When ds_salaries.csv is not present (primary data source).
    2. When it IS present, we generate 1,200 extra records to give the ML
       model enough training data (344 real records alone is too few for
       a model with 45 features).

    Experience years are biased by seniority: Principal/Staff roles get 8-15
    years, Senior/Lead get 4-10 years, everything else gets 0-12 years.
    """
    jobs = []
    today = datetime.now()
    for _ in range(n):
        title = random.choice(TITLES)
        sal_min, sal_max, sal_avg = _synthetic_salary(title)

        # More senior roles require more skills
        n_skills = (
            random.randint(4, 10)
            if any(x in title for x in ["Senior", "Staff", "Principal"])
            else random.randint(3, 7)
        )

        # Assign experience years based on seniority level
        experience = random.randint(0, 12)
        if any(x in title for x in ["Principal", "Staff"]):
            experience = random.randint(8, 15)
        elif any(x in title for x in ["Senior", "Lead"]):
            experience = random.randint(4, 10)

        location = random.choice(LOCATIONS)
        # 35% chance of remote regardless of listed location; always remote if location is "Remote"
        remote   = 1 if location == "Remote" or random.random() < 0.35 else 0

        days_ago = random.randint(0, 60)
        jobs.append({
            "title":       title,
            "company":     random.choice(COMPANIES),
            "location":    location,
            "salary_min":  sal_min,
            "salary_max":  sal_max,
            "salary_avg":  sal_avg,
            "skills":      json.dumps(_skills_for_title(title)),
            "experience":  experience,
            "job_type":    random.choice(["Full-time", "Contract", "Part-time"]),
            "remote":      remote,
            "date_posted": (today - timedelta(days=days_ago)).strftime("%Y-%m-%d"),
            "source":      "synthetic",
        })
    return jobs


# ── Database insert ───────────────────────────────────────────────────────────

def insert_jobs(jobs: list[dict]):
    """
    Inserts all job records into the SQLite database in a single batch.
    executemany() is much faster than calling execute() in a loop because
    it wraps all inserts in one transaction.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.executemany("""
        INSERT INTO jobs
            (title, company, location, salary_min, salary_max, salary_avg,
             skills, experience, job_type, remote, date_posted, source)
        VALUES
            (:title, :company, :location, :salary_min, :salary_max, :salary_avg,
             :skills, :experience, :job_type, :remote, :date_posted, :source)
    """, jobs)
    conn.commit()
    conn.close()
    print(f"Inserted {len(jobs)} records into the database.")


# ── Main entry point ──────────────────────────────────────────────────────────

def run():
    """
    Orchestrates the full data collection process:
    1. Initialize (or reset) the database
    2. Load real data from Kaggle if available, otherwise use synthetic
    3. If real data is used, supplement with synthetic records for ML training
    4. Insert everything into the database
    """
    init_db()

    if os.path.exists(CSV_PATH):
        print(f"Kaggle dataset found -> {CSV_PATH}")
        print("Loading real salary data...")
        real_jobs = load_from_kaggle(CSV_PATH)
        print(f"  {len(real_jobs)} real records loaded.")

        # The Kaggle dataset only gives us ~344 US records after filtering.
        # A model with 45 features needs more training data than that to learn
        # meaningful patterns without overfitting. We add 1,200 synthetic records
        # to supplement. The source column distinguishes them in the database.
        print("Generating synthetic records to supplement ML training...")
        synth_jobs = generate_synthetic(1200)
        jobs = real_jobs + synth_jobs
        print(f"  {len(synth_jobs)} synthetic records added.")
        print(f"  Total: {len(jobs)} records ({len(real_jobs)} real + {len(synth_jobs)} synthetic).")
    else:
        print("=" * 62)
        print("  Kaggle dataset not found. To use real data:")
        print("  1. Visit: kaggle.com/datasets/ruchi798/data-science-job-salaries")
        print("  2. Download ds_salaries.csv")
        print("  3. Place it at: data/ds_salaries.csv")
        print("  4. Re-run: python pipeline/collector.py")
        print("=" * 62)
        print("Using synthetic data (2,000 records)...")
        jobs = generate_synthetic(2000)

    insert_jobs(jobs)
    print("Data collection complete.")


if __name__ == "__main__":
    run()
