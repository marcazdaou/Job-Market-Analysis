#!/usr/bin/env python3
"""
Job Market Analytics - One-click setup script

Runs the entire pipeline in order:
  1. Install Python dependencies from requirements.txt
  2. Load data and populate the SQLite database (collector.py)
  3. Train and compare ML models, save the best one (train.py)
  4. Launch the Streamlit dashboard in your browser

Usage:
    python setup.py
"""

import subprocess
import sys
import os

# Get the absolute path of the folder this file lives in.
# We use this as the working directory so all relative paths in subprocesses work correctly.
BASE = os.path.dirname(os.path.abspath(__file__))


def run(cmd, cwd=None):
    """
    Runs a shell command as a subprocess and exits the script if it fails.
    sys.executable gives the path to the current Python interpreter,
    which ensures we use the same Python/environment for all steps.
    """
    print(f"\n Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd or BASE)
    if result.returncode != 0:
        # If any step fails (e.g. pip install fails, or collector crashes),
        # we stop immediately rather than continuing to the next step.
        print(f"  Failed: {' '.join(cmd)}")
        sys.exit(1)


def main():
    print("=" * 55)
    print("  Job Market Analytics - Setup")
    print("=" * 55)

    # Step 1: Install all required Python packages.
    # The -q flag suppresses verbose pip output to keep things readable.
    print("\n[1/3] Installing dependencies...")
    run([sys.executable, "-m", "pip", "install", "-q",
         "streamlit", "plotly", "pandas", "scikit-learn",
         "numpy", "requests"])

    # Step 2: Run the data collector.
    # This creates data/jobs.db and populates it with job records.
    # If data/ds_salaries.csv exists, it uses real Kaggle data.
    # Otherwise it falls back to synthetic data.
    print("\n[2/3] Loading job market data into database...")
    run([sys.executable, "pipeline/collector.py"])

    # Step 3: Train the salary prediction model.
    # Reads from data/jobs.db, compares 3 models, saves the best one to
    # models/salary_model.pkl and models/model_meta.json.
    print("\n[3/3] Training salary prediction model...")
    run([sys.executable, "models/train.py"])

    print("\n" + "=" * 55)
    print("  Setup complete. Launching dashboard...")
    print("  Open http://localhost:8501 in your browser")
    print("=" * 55 + "\n")

    # Step 4: Launch the Streamlit dashboard.
    # --server.headless true prevents Streamlit from opening a browser tab
    # automatically on some systems. You open it manually at localhost:8501.
    run([sys.executable, "-m", "streamlit", "run",
         "dashboard/app.py", "--server.headless", "true"])


if __name__ == "__main__":
    main()
