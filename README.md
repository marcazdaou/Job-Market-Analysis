# Job Market Analytics Dashboard

An end-to-end Python data analytics and machine learning project for exploring a **demonstration dataset** of data engineering and analytics roles. The project builds a SQLite database, compares salary-prediction models, and presents interactive visualizations and a salary estimator in Streamlit.

> **Data transparency:** This is a portfolio and learning project, **not a live job board or a source of current hiring statistics**. The pipeline can use historical salary records from Kaggle supplemented with synthetic records, or generate a fully synthetic dataset when the Kaggle file is absent. Company names, skill requirements, and posting dates are generated for demonstration; they do not represent verified vacancies at the named companies.

## Overview

The workflow has four stages:

1. **Data preparation:** Load eligible U.S. salary records from an optional Kaggle CSV, or generate synthetic fallback records; create additional synthetic records when the CSV is present.
2. **Storage:** Rebuild and populate a local SQLite `jobs` table with source labels distinguishing Kaggle-derived and synthetic records.
3. **Machine learning:** Engineer numerical features, evaluate three regression algorithms, and save the selected model and its evaluation metadata.
4. **Visualization:** Run a Streamlit dashboard with interactive charts, filters, model comparisons, and a salary estimator.

## Features

### Market-analysis dashboard

- Summary cards for dataset record count, average salary, remote share, and distinct company labels.
- Salary by job title, skill frequency and salary associations, location comparisons, salary distribution, posting-date distribution, and experience-based visualizations.
- Filters for job title, location, remote/on-site status, salary range, and date range.

**Interpretation:** The cards and charts describe the records currently loaded into SQLite. In synthetic-only mode, all values are simulated. In mixed mode, only the source salary values and other fields carried over from Kaggle are historical source data; generated companies, skills, dates, and estimated salary ranges are not independently observed job postings. The “companies” metric counts assigned company labels, not confirmed employers hiring.

### Salary estimator

- Accepts job title, location, experience, and skill inputs.
- Uses the saved model to produce a salary estimate and illustrative range.
- Highlights additional skills using the project's dataset-driven logic.

Estimates are demonstrations of the trained model, **not compensation advice or verified market quotes**.

### Model comparison

- Compares **Linear Regression**, **Random Forest**, and **Gradient Boosting**.
- Uses an **80/20 train/test split** and **5-fold cross-validation**.
- Displays test mean absolute error (MAE), test R², and mean cross-validation R².
- Saves the model selected by the highest mean cross-validation R², along with its feature-column order and evaluation metadata.

## Technology

| Area | Tools |
| --- | --- |
| Language | Python 3.11+ |
| Data preparation | Pandas, NumPy |
| Database | SQLite |
| Machine learning | scikit-learn |
| Dashboard and charts | Streamlit, Plotly |
| Model artifacts | Pickle, JSON |

## Data sources and limitations

The data collector (`pipeline/collector.py`) supports two modes:

| Mode | Records loaded | What the records represent |
| --- | --- | --- |
| Kaggle CSV available | Eligible U.S. Kaggle salary records **plus 1,200 synthetic records** | Historical salary records supplemented by simulated job records. The number of eligible Kaggle rows depends on the CSV and filtering. |
| Kaggle CSV unavailable | **2,000 synthetic records** | A fully simulated dataset for running the complete project without the optional CSV. |

The collector filters Kaggle records to `company_location == "US"` and `salary_in_usd >= 50000`. The previously documented **344 Kaggle records** is an example from one dataset version and run, not a guaranteed count. With 344 eligible Kaggle rows, mixed mode would contain **1,544 total records**.

The Kaggle file contains salary data, **not verified current job advertisements**. The pipeline assigns company names and skills, generates posting dates within the preceding 60 days, estimates salary bounds around Kaggle salary values, and generates experience-year values from experience bands. In synthetic-only mode, salaries and the other job fields are simulated. Accordingly, charts titled “postings,” “companies hiring,” or “in-demand skills” should be understood as descriptions of the demonstration dataset rather than measured hiring-market activity.

Kaggle dataset: [Data Science Job Salaries](https://www.kaggle.com/datasets/ruchi798/data-science-job-salaries). Download the dataset yourself if you want to use the optional CSV.

## Project structure

```text
Job-Market-Analysis/
├── pipeline/
│   └── collector.py       # Load or generate records and populate SQLite
├── models/
│   ├── train.py           # Feature engineering, evaluation, model selection
│   ├── salary_model.pkl   # Generated trained model
│   └── model_meta.json    # Generated evaluation metadata
├── dashboard/
│   └── app.py             # Streamlit dashboard
├── data/
│   ├── ds_salaries.csv    # Optional Kaggle CSV supplied by the user
│   └── jobs.db            # Generated SQLite database
├── setup.py               # Install, prepare data, train, launch dashboard
├── requirements.txt
└── README.md
```

Generated files may not exist until you run the pipeline.

## Run locally

**Requirements:** Python 3.11 or newer and pip.

### 1. Clone the repository

```bash
git clone https://github.com/marcazdaou/Job-Market-Analysis.git
cd Job-Market-Analysis
```

### 2. Optionally add the Kaggle dataset

Download `ds_salaries.csv` from the [Kaggle dataset page](https://www.kaggle.com/datasets/ruchi798/data-science-job-salaries) and place it at `data/ds_salaries.csv`.

If you skip this step, the collector generates 2,000 synthetic records. The dataset mode is determined when the collector runs; adding the CSV later requires rerunning the collector and retraining the model.

### 3. Run the setup script

```bash
python setup.py
```

The script installs its Python dependencies, rebuilds the SQLite database, trains the models, and starts Streamlit. Open **http://localhost:8501** if your browser does not open automatically.

**Alternative: run each stage manually**

```bash
python -m pip install -r requirements.txt
python pipeline/collector.py
python models/train.py
python -m streamlit run dashboard/app.py
```

**Note:** Running the collector drops and recreates the `jobs` table. Do not store records you need to preserve in the generated database without backing them up first.

## Model training and evaluation

`models/train.py` engineers features from experience, remote status, job-title seniority, skill count, **27 individual skill indicators**, one-hot-encoded locations, and employment types. It trains the three candidate regressors and records test MAE, test R², and mean and standard deviation of 5-fold cross-validation R².

The selected model is saved to `models/salary_model.pkl`; the comparison results, feature count, and selected-model metrics are saved to `models/model_meta.json`.

**For exact performance numbers, use the output of your own run or the generated `model_meta.json`.** Scores vary with the input CSV, randomly generated records, and the available features. Do not treat a score obtained on fully synthetic records as evidence of accuracy on real-world salaries. The current training script computes cross-validation on the full feature matrix, including the held-out test portion; consequently, its cross-validation score is useful for this project’s comparison but should not be presented as a fully independent validation result.

## Known limitations

- The project does not scrape or continuously refresh real job advertisements.
- Company names and posting dates are assigned, not verified; the date filter does not establish current market trends.
- Skill frequencies and skill-related salary comparisons reflect generated skill assignments.
- Synthetic data can make model performance look different from performance on real job-market data.
- The model and dashboard are educational demonstrations, not tools for hiring decisions or salary negotiation.

## Author

**Marc Abi Zeid Daou**  
[GitHub](https://github.com/marcazdaou)
