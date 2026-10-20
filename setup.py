#!/usr/bin/env python3
"""Rebuild everything from the raw data, then launch the dashboard.

Not needed to *view* the project: the cleaned data and trained model are
committed, so `streamlit run dashboard/app.py` (or the hosted link) works as is.

Steps:
  1. Install the pinned dependencies from requirements.txt
  2. Clean the raw postings      -> data/jobs_clean.parquet   (pipeline/collector.py)
  3. Train and evaluate the model -> models/salary_model.pkl  (models/train.py)
  4. Launch the dashboard at http://localhost:8501

Usage:  python setup.py
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(args):
    """Run one step with the current interpreter; stop on the first failure."""
    print(f"\n> {' '.join(args)}", flush=True)
    if subprocess.run([sys.executable, *args], cwd=ROOT).returncode != 0:
        sys.exit(f"Step failed: {' '.join(args)}")


def main():
    run(["-m", "pip", "install", "-q", "-r", "requirements.txt"])
    run(["pipeline/collector.py"])
    run(["models/train.py"])
    run(["-m", "streamlit", "run", "dashboard/app.py"])


if __name__ == "__main__":
    main()
