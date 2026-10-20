# Job Market Analytics

An end-to-end analysis of **394,300 job postings across every type of role** (software, data, sales,
engineering, operations, healthcare, retail and more) from 6,295 companies in 65 countries, with a
machine-learning salary model that explains **77% of the variance in advertised salaries** on held-out data.

**Which jobs and which salaries?** Tech roles (software, data, product, security) are 21% of postings; the
other 79% span every other field. Salary figures use the 64,574 postings that list a yearly USD salary;
82% of those are US jobs, 31% are tech jobs and 45% are senior level or above, so they describe these
postings rather than national averages.

**[Open the live dashboard →](https://job-market-analysis-ndmqelnezlfa64hyyhhwkv.streamlit.app)**
Everything is precomputed and committed, so the dashboard opens with no setup.

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.58-red?logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-orange?logo=scikit-learn)
![Plotly](https://img.shields.io/badge/Plotly-6.7-purple?logo=plotly)

---

## The questions, and what the data says

| # | Question | Answer |
|---|---|---|
| 1 | How much do these jobs pay? | Typical (median) advertised salary is **$160k**; half fall between $113k and $206k (64,574 postings with a yearly USD salary). |
| 2 | Which jobs pay the most? | **Software Engineering ($195k), Data & AI ($191k), Product & Design ($190k)**, then Security ($188k). |
| 3 | How much does seniority matter? | **2.7×** from entry level ($83k) to principal/staff ($223k); Senior/Lead $170k, Director/Executive $204k. |
| 4 | Do skills raise pay? | Jobs asking for PyTorch pay $220k typically, but once title, company and city are accounted for, **a single skill shifts pay by at most 4%**. The job and the employer matter far more. |
| 5 | Are employers open about pay? | **27%** of postings show a salary, driven by law: 56% in the US and 27% in the UK, but 3% in Germany and 1% in France. |
| 6 | Can machine learning predict a salary? | **Yes:** on 10,972 unseen jobs the model explains 77% of pay differences (R² 0.77), with a typical error of 11%. |

The dashboard opens with these six answers and lets you explore each one interactively.

---

## Dashboard

| Tab | What it shows |
|---|---|
| **Overview** | The six key findings, headline numbers, an interactive world map of postings and pay, in-demand job types, and a "who hires vs. who pays" company chart |
| **Salaries** | Salary distribution with a "compare your salary" marker; typical pay and spread by job type, seniority, city and industry; remote vs. on-site pay gap |
| **Skills** | Skill explorer (pick a skill → demand, pay, paired skills, roles); most requested skills; best-paid skills; ML estimate of each skill's own effect on pay |
| **Explore a role** | Search any job title, or compare two side by side: pay distribution, skills, best-paying companies, and the latest postings with links |
| **Salary predictor (ML)** | Describe a job → estimated salary, likely range, and where it sits among similar real jobs; full model evaluation |
| **About the data** | Source, preparation steps, model method, and a searchable table of postings |

Sidebar filters (role family, seniority, country, remote/on-site, employment type) apply to every tab
except the predictor. Tooltips add context (pay, remote share, top skill) rather than repeating labels,
and every chart has a "View data table" option.

---

## Data

**Source:** `zalizedata/tech-job-postings-salary-dataset` (CC BY-NC 4.0): job postings collected from
public company career pages (SmartRecruiters, Greenhouse, Ashby, Lever, Workable), covering roles open
in August 2026.

**Cleaning** (`pipeline/collector.py`, rules in `pipeline/features.py`):

| Salary status | Postings |
|---|---:|
| No salary listed | 287,149 |
| **Usable annual USD salary** | **64,574** |
| Another currency, or hourly/monthly pay | 42,029 |
| Range wider than 4× (uninformative) | 303 |
| Midpoint outside $20k–$1M (hourly rate mislabeled as annual, or typo) | 200 |
| Invalid range (min ≤ 0 or max < min) | 45 |

- All postings count toward volumes, role mix and skill demand. Salary statistics use only the 64,574
  usable annual USD ranges; other currencies and pay periods are **excluded, not converted**, so no
  exchange-rate or hours-per-year assumption is introduced.
- Salary = midpoint of the advertised range. Medians are reported because pay is right-skewed.
- **Role family** (19 job types) and **seniority** (5 levels) are inferred from job-title keywords using
  transparent, ordered regex rules, with French, German, Spanish and Dutch keywords for non-English
  titles. 86% of postings (96% of those with a salary) match a job type; the rest have very generic or
  less common-language titles and are left out of job-type charts only.

---

## Machine learning

**Task:** predict the advertised annual salary of a posting.
**Target:** `log(salary midpoint)`, so errors are proportional (a $20k miss matters more at $60k than at $400k).

**Features:** title text (TF-IDF unigrams + bigrams), company, industry, department, city, country,
employment type, remote flag, skills (binary flags), and title-derived seniority / role family.

**Evaluation protocol (designed to avoid leakage):**
1. The same job is often posted in many cities with an identical salary range. These copies are collapsed
   (64,574 → 54,859) so a test row can never be a duplicate of a training row.
2. 20% is held out as a test set and not touched until the end.
3. Candidates are compared with 5-fold cross-validation on the remaining 80%; the lowest CV error wins.
4. The winner is scored once on the test set, and again on a split where **entire companies** are held out.
5. An 80% prediction interval is built from out-of-fold residuals (split-conformal style) and its
   coverage is checked on the test set.

**Results** (dollar scale):

| Model | CV R² | Test R² | Test MAE |
|---|---:|---:|---:|
| Median baseline | −0.004 | −0.003 | $55,869 |
| Ridge: title + location only (previous version of this project) | 0.586 | 0.611 | $31,437 |
| Gradient boosting (target-encoded categoricals) | 0.645 | 0.676 | $28,663 |
| **Ridge: full feature set (selected)** | **0.737** | **0.771** | **$24,050** |

| Metric | Value |
|---|---:|
| Test R² (log scale) | 0.806 |
| Median absolute % error | 11.1% |
| 80% interval coverage on test set | 80.9% |
| R² on companies never seen in training | 0.496 |

The linear model beats gradient boosting here because most of the signal sits in thousands of sparse
title words and company identities, which a regularized linear model handles directly. Stacking both
models was tested and added only +0.006 R², so the simpler, interpretable model was kept.

**Permutation importance** (drop in test R² when one input is shuffled): title 0.58, company 0.40,
department 0.06, industry 0.05, city 0.03, skills 0.01.

**Scope:** salaries are the ranges employers advertise; most salaried postings are from the US, where
pay-transparency laws require ranges; role family and seniority come from title keywords; skill–pay
relationships are associations rather than cause and effect; for employers outside the training data the
model scores R² 0.50; jobs that usually pay by the hour (retail, hospitality, trades) have few yearly
salaries, so the predictor flags its estimates for them as less reliable; pay gaps between groups (e.g.
remote vs. on-site) can partly reflect a different mix of jobs in each group.

---

## Project structure

```
├── pipeline/
│   ├── features.py        # Shared rules: salary cleaning, role family, seniority, model inputs
│   └── collector.py       # Step 1: raw parquet -> data/jobs_clean.parquet + data_summary.json
├── models/
│   ├── train.py           # Step 2: compare models, evaluate, save the winner
│   ├── salary_model.pkl   # Trained model (committed)
│   └── model_meta.json    # Metrics, comparison, chart data (committed)
├── dashboard/
│   └── app.py             # Step 3: Streamlit dashboard (reads committed artifacts only)
├── data/
│   ├── jobs_clean.parquet # Cleaned postings, 11 MB (committed)
│   └── data_summary.json  # Cleaning funnel (committed)
├── .streamlit/config.toml # Dashboard theme
├── check_data.py          # Quick profile of the raw file
├── setup.py               # Rebuild everything from raw data and launch
└── requirements.txt       # Pinned versions (the model pickle is version-specific)
```

---

## Run locally

```bash
git clone https://github.com/marcazdaou/Job-Market-Analysis.git
cd Job-Market-Analysis
pip install -r requirements.txt
streamlit run dashboard/app.py
```

**To rebuild from raw data:** place the source file at `data/tech_jobs_2026/jobs_large.parquet`, then run
`python setup.py` (or `python pipeline/collector.py` followed by `python models/train.py`). Training takes about 2 minutes on a laptop.

---

## Author

**Marc Abi Zeid Daou**, BS Computer Science, UMass Boston
[LinkedIn](https://www.linkedin.com/in/marc-abi-zeid-daou/) · [GitHub](https://github.com/marcazdaou) · [Portfolio](https://marcazdaou.github.io/my-portfolio/)
