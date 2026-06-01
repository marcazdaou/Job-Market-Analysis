# Job Market Analytics Dashboard

A full-stack data analytics platform for exploring Data Engineering and Analytics job market trends. Built with Python, Streamlit, and Scikit-learn — covering the full pipeline from data ingestion to machine learning to interactive visualization.

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red?logo=streamlit)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4+-orange?logo=scikit-learn)
![Plotly](https://img.shields.io/badge/Plotly-5.19+-purple?logo=plotly)
![SQLite](https://img.shields.io/badge/SQLite-3-lightblue?logo=sqlite)

---

## Overview

This project demonstrates an end-to-end data engineering and analytics workflow:

1. **Data ingestion** — Loads real salary data from the Kaggle Data Science Job Salaries dataset, supplemented with synthetic records for ML training
2. **Storage** — Persists all records in a structured SQLite database
3. **Machine learning** — Compares three regression models (Linear Regression, Random Forest, Gradient Boosting) using cross-validation and selects the best performer for salary prediction
4. **Visualization** — Serves an interactive Streamlit dashboard with filters, charts, and a salary estimator

---

## Features

**Market Analysis**
- KPI cards: total postings, average salary, remote percentage, companies hiring
- Average salary by job title
- Top in-demand skills across all postings
- Salary premium per skill — which technologies correlate with higher pay
- Postings by location, colored by average salary
- Salary distribution histogram
- Posting volume over time
- Experience vs. salary scatter plot
- Salary by experience band (SQL-aggregated)

**Salary Estimator**
- Input a job title, location, years of experience, and skills
- Returns a predicted salary with conservative and optimistic ranges
- Identifies high-demand skills missing from the input profile

**Model Comparison**
- Trains and compares Linear Regression, Random Forest, and Gradient Boosting
- Evaluates all three using 5-fold cross-validation
- Displays results as a bar chart and formatted table
- Selects the model with the highest cross-validated R²

**Interactive Filters**
- Filter by job title, location, work type (remote / on-site)
- Salary range slider
- Date range control

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| Data Pipeline | Pandas, SQLite |
| Machine Learning | Scikit-learn (LinearRegression, RandomForest, GradientBoosting) |
| Dashboard | Streamlit, Plotly |
| Data Storage | SQLite (jobs.db) |
| Feature Engineering | One-hot encoding, seniority extraction, skill flags |

---

## Project Structure

```
job-market-analytics/
├── pipeline/
│   └── collector.py          # Data ingestion and SQLite loading
├── models/
│   ├── train.py              # Model training and comparison
│   ├── salary_model.pkl      # Saved model (auto-generated)
│   └── model_meta.json       # Model metrics (auto-generated)
├── dashboard/
│   └── app.py                # Streamlit dashboard
├── data/
│   ├── ds_salaries.csv       # Kaggle dataset (you provide this)
│   └── jobs.db               # SQLite database (auto-generated)
├── setup.py                  # One-command setup script
├── requirements.txt
└── README.md
```

---

## Setup

### Requirements

- Python 3.11 or higher
- pip

### Step 1 — Clone the repository

```bash
git clone https://github.com/marcazdaou/job-market-analytics.git
cd job-market-analytics
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — Add the Kaggle dataset (optional but recommended)

1. Go to [kaggle.com/datasets/ruchi798/data-science-job-salaries](https://www.kaggle.com/datasets/ruchi798/data-science-job-salaries)
2. Create a free Kaggle account if you do not have one
3. Download `ds_salaries.csv`
4. Place it at `data/ds_salaries.csv`

If you skip this step, the app will run using synthetic data only.

### Step 4 — Run the setup script

```bash
python setup.py
```

This will:
- Load the dataset and populate the SQLite database
- Train and compare the three regression models
- Launch the dashboard at [http://localhost:8501](http://localhost:8501)

### Manual steps (alternative to setup.py)

```bash
python pipeline/collector.py   # Load data into database
python models/train.py         # Train and compare models
streamlit run dashboard/app.py # Launch dashboard
```

---

## Model Performance

Three models are trained and evaluated on every run. Results from the current dataset:

| Model | Test MAE | Test R2 | CV R2 |
|---|---|---|---|
| Linear Regression | ~$18,700 | 0.43 | 0.52 |
| Random Forest | ~$16,500 | 0.48 | 0.57 |
| Gradient Boosting | ~$17,200 | 0.44 | 0.50 |

The model with the highest cross-validated R2 is automatically selected. Results will vary slightly on each run due to random data generation.

**Features used:** seniority level, years of experience, remote flag, skill count, individual skill flags (27 technologies), location (one-hot), employment type (one-hot)

---

## Data Sources

| Records | Source |
|---|---|
| 344 real records | Kaggle — Data Science Job Salaries (self-reported, 2020-2022) |
| 1,200 synthetic records | Generated to supplement ML training |

Company names, skills, and posting dates are synthetically assigned. Salary figures for the real records come directly from the Kaggle dataset.

---
PS: The model's R² reflects the limitations of the available features. Salary in this industry is largely driven by company, year, and negotiation — factors not captured in job postings. The model still captures the directional signal from seniority and location, which is the best you can do with this dataset.

## Author

**Marc Abi Zeid Daou**
BS Computer Science, UMass Boston

[LinkedIn](https://www.linkedin.com/in/marc-abi-zeid-daou/) · [GitHub](https://github.com/marcazdaou) · [Portfolio](https://marcazdaou.github.io)
