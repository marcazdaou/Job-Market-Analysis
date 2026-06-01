"""
Job Market Analytics - Salary Prediction Model

This script is the second step in the pipeline. It reads the job data from
SQLite, engineers features the model can learn from, trains three different
regression models, compares them using cross-validation, and saves the best one.

Run after collector.py:
    python models/train.py
"""

import sqlite3
import json
import os
import pickle
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score

# ── File paths ────────────────────────────────────────────────────────────────
DB_PATH    = os.path.join(os.path.dirname(__file__), "../data/jobs.db")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/salary_model.pkl")
META_PATH  = os.path.join(os.path.dirname(__file__), "../models/model_meta.json")

# ── Skills to create binary features for ─────────────────────────────────────
# For each skill listed here, the model gets a 0/1 column per record:
# 1 if the job requires that skill, 0 if it does not.
# This lets the model learn that e.g. "jobs requiring Kubernetes pay more".
TOP_SKILLS = [
    "Python", "SQL", "Spark", "Kafka", "Airflow", "dbt", "Snowflake",
    "AWS", "GCP", "Azure", "Docker", "Kubernetes", "Terraform",
    "Pandas", "Scikit-learn", "TensorFlow", "PyTorch", "Scala",
    "Tableau", "Power BI", "Databricks", "Machine Learning", "Deep Learning",
    "Statistics", "Data Modeling", "ETL", "Data Warehousing",
]

# ── Candidate models ──────────────────────────────────────────────────────────
# We train all three and let cross-validation decide which is best.
# - Linear Regression: fast, simple, assumes salary is a linear combination of features
# - Random Forest: builds many decision trees and averages them, handles non-linear patterns
# - Gradient Boosting: builds trees sequentially, each correcting the previous one's errors
CANDIDATE_MODELS = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=100,   # number of trees in the forest
        random_state=42,    # fixed seed so results are reproducible
        n_jobs=-1           # use all CPU cores to train faster
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=300,   # number of boosting stages (trees)
        learning_rate=0.05, # how much each tree corrects the previous one (small = more careful)
        max_depth=5,        # maximum depth of each tree (limits overfitting)
        subsample=0.8,      # use 80% of data per tree (adds randomness, reduces overfitting)
        random_state=42,
    ),
}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    """
    Reads all records from the jobs table into a pandas DataFrame.
    Using read_sql lets us run a SQL query and get the result as a DataFrame
    directly, without manually looping over rows.
    """
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM jobs", conn)
    conn.close()
    return df


# ── Feature engineering ───────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts raw text columns into numeric features the ML model can use.
    ML models only understand numbers — they cannot process strings like
    "San Francisco, CA" or "Full-time" directly.

    Features created:
    1. Location columns (loc_*)     — one-hot encoding of top 10 cities
    2. Job type columns (type_*)    — one-hot encoding of employment types
    3. Seniority (1-4)              — extracted from keywords in the job title
    4. Skill columns (skill_*)      — binary flags for 27 key technologies
    5. skill_count                  — total number of skills required
    """

    # --- Location one-hot encoding ---
    # Find the 10 most common locations in the dataset.
    # For each one, create a new column that is 1 if this record is in that
    # location and 0 otherwise. This is called "one-hot encoding".
    # We use "safe" column names by replacing spaces and commas with underscores.
    top_locs = df["location"].value_counts().nlargest(10).index.tolist()
    for loc in top_locs:
        safe = loc.replace(" ", "_").replace(",", "")
        df[f"loc_{safe}"] = (df["location"] == loc).astype(int)

    # --- Job type one-hot encoding ---
    # Same idea as locations: create a column for each unique job_type value.
    for jt in df["job_type"].unique():
        df[f"type_{jt.replace('-', '_')}"] = (df["job_type"] == jt).astype(int)

    # --- Seniority score ---
    # Converts the job title into a numeric seniority level (1-4).
    # The model can then learn that higher seniority = higher salary.
    def seniority(title):
        if any(x in title for x in ["Principal", "Staff"]):
            return 4   # most senior
        if any(x in title for x in ["Senior", "Lead"]):
            return 3
        if any(x in title for x in ["Junior", "Associate"]):
            return 1   # most junior
        return 2       # default: mid-level

    df["seniority"] = df["title"].apply(seniority)

    # --- Skill binary flags ---
    # For each skill in TOP_SKILLS, create a column that is 1 if that skill
    # appears in the job's skills JSON list, and 0 if it does not.
    # The lambda sk=skill captures the current skill in the loop so it doesn't
    # get overwritten as the loop continues (a common Python gotcha).
    def has_skill(skills_json, skill):
        try:
            return int(skill in json.loads(skills_json))
        except Exception:
            return 0  # if the JSON is malformed, treat as no skill

    for skill in TOP_SKILLS:
        col = f"skill_{skill.replace(' ', '_').replace('-', '_')}"
        df[col] = df["skills"].apply(lambda s, sk=skill: has_skill(s, sk))

    # --- Total skill count ---
    # A simple numeric summary: how many skills does this job require in total?
    df["skill_count"] = df["skills"].apply(
        lambda s: len(json.loads(s)) if s else 0
    )

    return df


# ── Model training and comparison ─────────────────────────────────────────────

def train():
    """
    Full training pipeline:
    1. Load data from SQLite
    2. Engineer features
    3. Split into train/test sets
    4. Train each candidate model and evaluate with cross-validation
    5. Select the best model by CV R² score
    6. Save the best model and all metrics to disk
    """
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    print("Loading data...")
    df = load_data()
    print(f"  {len(df)} records loaded.")

    print("Engineering features...")
    df = build_features(df)

    # Build the final list of feature column names.
    # These are the columns the model will actually use as input.
    # X = the input features, y = the target (salary) we want to predict.
    feature_cols = (
        ["experience", "remote", "seniority", "skill_count"]  # base numeric features
        + [c for c in df.columns if c.startswith("loc_")]     # one-hot location columns
        + [c for c in df.columns if c.startswith("type_")]    # one-hot job type columns
        + [c for c in df.columns if c.startswith("skill_")]   # binary skill flag columns
    )

    X = df[feature_cols].fillna(0)  # fill any missing values with 0
    y = df["salary_avg"]            # what we are trying to predict

    # Split data: 80% for training, 20% for testing.
    # random_state=42 means the split is the same every time you run this,
    # so results are reproducible and comparable.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"\nComparing {len(CANDIDATE_MODELS)} models (5-fold cross-validation)...")
    print(f"  {'Model':<25} {'Test MAE':>12} {'Test R2':>9} {'CV R2':>10}")
    print("  " + "-" * 58)

    comparison = {}    # stores metrics for all models (shown in dashboard)
    trained_models = {}  # stores fitted model objects so we can save the best one

    for name, model in CANDIDATE_MODELS.items():
        t0 = time.time()
        model.fit(X_train, y_train)       # train the model on the training set
        elapsed = time.time() - t0

        y_pred = model.predict(X_test)    # make predictions on the unseen test set
        mae = mean_absolute_error(y_test, y_pred)  # average dollar error
        r2  = r2_score(y_test, y_pred)             # % of variance explained (1.0 = perfect)

        # Cross-validation: train/test the model 5 different ways on different
        # subsets of data. This is more reliable than a single train/test split
        # because it tests whether the model generalizes, not just memorizes.
        cv  = cross_val_score(model, X, y, cv=5, scoring="r2")

        comparison[name] = {
            "mae":          round(mae, 2),
            "r2":           round(r2, 4),
            "cv_r2_mean":   round(float(cv.mean()), 4),   # average CV score
            "cv_r2_std":    round(float(cv.std()), 4),    # how much CV scores varied
            "train_time_s": round(elapsed, 2),
        }
        trained_models[name] = model
        print(
            f"  {name:<25} ${mae:>10,.0f} {r2:>9.3f}"
            f" {cv.mean():>8.3f} +/- {cv.std():.3f}"
            f"  ({elapsed:.1f}s)"
        )

    # Select the model with the highest cross-validated R².
    # We use CV R² rather than test R² because CV is more reliable —
    # it evaluates performance across multiple different subsets of the data.
    best_name = max(comparison, key=lambda k: comparison[k]["cv_r2_mean"])
    best_model = trained_models[best_name]
    print(f"\nSelected: {best_name}  (highest cross-validated R2)")

    # Feature importances show which inputs the model relied on most.
    # Only tree-based models (Random Forest, Gradient Boosting) have this —
    # Linear Regression uses coefficients instead, so we skip it for LR.
    top_features = {}
    if hasattr(best_model, "feature_importances_"):
        importances = pd.Series(best_model.feature_importances_, index=feature_cols)
        top_features = {k: round(v, 4) for k, v in importances.nlargest(15).to_dict().items()}

    # ── Save model ─────────────────────────────────────────────────────────────
    # We save both the model object and the feature column names together.
    # The feature_cols list is essential: the dashboard must build the input
    # vector in exactly the same column order the model was trained on,
    # otherwise the prediction will be wrong.
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": best_model, "model_name": best_name, "feature_cols": feature_cols}, f)

    # ── Save metadata ──────────────────────────────────────────────────────────
    # model_meta.json is read by the dashboard to display model performance
    # metrics and the model comparison chart without needing to re-run training.
    meta = {
        "best_model":       best_name,
        "mae":              comparison[best_name]["mae"],
        "r2":               comparison[best_name]["r2"],
        "cv_r2_mean":       comparison[best_name]["cv_r2_mean"],
        "cv_r2_std":        comparison[best_name]["cv_r2_std"],
        "n_features":       len(feature_cols),
        "n_train":          len(X_train),
        "model_comparison": comparison,    # all three models' metrics for the comparison chart
        "top_features":     top_features,  # empty dict if best model is Linear Regression
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nModel saved  -> {MODEL_PATH}")
    print(f"Metadata     -> {META_PATH}")
    return meta


if __name__ == "__main__":
    train()
