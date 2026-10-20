"""How reliable are the job-type labels? Check them against the employer's own department.

Job types (role families) are inferred from title keywords. Many postings also
carry a department chosen by the employer (e.g. "Engineering", "Sales",
"People"). That department is independent of the title rules, so it is a
natural check: if a posting is labeled "Sales & Business Dev.", the employer
should usually have put it in a sales-type department.

Method
  1. Map clear department names to a department group with keyword rules
     (ambiguous names such as "Operations", "Global" or store names are skipped).
  2. Each department group lists the job types that plausibly belong in it
     (an "Engineering" department contains software, hardware, data and
     security jobs).
  3. Agreement = share of postings whose job type is compatible with their
     department group, for job types tied to one function. Cross-functional job
     types (Operations & Admin, Consulting & Strategy, Management) legitimately
     sit in every kind of department, so a department cannot confirm them; they
     are reported but not scored. "Other" and "Students & Apprentices" are not
     functions and are also not scored.

Output: analysis/results/label_validation.json (read by the dashboard, README and report)
Run from the project root:  python analysis/label_validation.py
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "jobs_clean.parquet"
OUT = ROOT / "analysis" / "results" / "label_validation.json"
CROSS_FUNCTIONAL = {"Operations & Admin", "Consulting & Strategy", "Management"}
EXCLUDED = CROSS_FUNCTIONAL | {"Other", "Students & Apprentices"}

SWE, HW, DATA_AI, SEC = "Software Engineering", "Hardware & Engineering", "Data & AI", "Security"
PRODUCT, RESEARCH, SALES = "Product & Design", "Research & Science", "Sales & Business Dev."
MARKETING, SUPPORT, FIN = "Marketing & Comms", "Customer Success & Support", "Finance & Legal"
HR, HEALTH, RETAIL = "People & HR", "Healthcare", "Retail & Hospitality"
EDUCATION, TRADES = "Education", "Skilled Trades & Field"

# (department group, keyword pattern, compatible job types). First match wins, so
# specific groups come before the broad "Engineering & technology" group.
DEPARTMENT_RULES = [
    ("Data", r"\b(data|analytics|machine learning|ai)\b", {DATA_AI, SWE, RESEARCH}),
    ("Security", r"\b(security|cyber\w*)\b", {SEC, SWE}),
    ("Sales", r"\b(sales|business development|revenue|gtm|go to market|commercial|partnerships?|"
              r"account management)\b", {SALES, SUPPORT, MARKETING}),
    ("Marketing", r"\b(marketing|communications?|brand|growth|content)\b", {MARKETING, PRODUCT, SALES}),
    ("Customer support", r"\b(customer|client services|support)\b", {SUPPORT, SALES}),
    ("Finance & legal", r"\b(finance|accounting|legal|tax|treasury|compliance)\b", {FIN}),
    ("People & HR", r"\b(people|human resources|hr|talent|recruiting)\b", {HR}),
    ("Healthcare", r"\b(clinical|medical|nursing|health\w*|patient care|soins|telehealth)\b", {HEALTH, RESEARCH}),
    ("Retail & stores", r"\b(retail|stores?|restaurants?)\b", {RETAIL, SALES}),
    ("Education", r"\b(education|teach\w*|school|tutoring|academic)\b", {EDUCATION}),
    ("Field & trades", r"\b(inspection|field (testing|services|operations)|maintenance|construction|"
                       r"installation|technicians?)\b", {TRADES, HW}),
    ("Research", r"\b(research|science|laboratory)\b", {RESEARCH, DATA_AI, HEALTH}),
    ("Product & design", r"\b(product|design|ux)\b", {PRODUCT, SWE, DATA_AI}),
    ("Engineering & technology", r"\b(engineering|software|development|r&d|technology|tech|it|infrastructure|"
                                 r"platform)\b", {SWE, HW, DATA_AI, SEC, PRODUCT, RESEARCH}),
]
_COMPILED = [(name, re.compile(rx), ok) for name, rx, ok in DEPARTMENT_RULES]


def department_group(department) -> str | None:
    """Department name -> department group, or None when the name is missing or ambiguous."""
    if not isinstance(department, str):
        return None
    text = department.lower()
    return next((name for name, pattern, _ in _COMPILED if pattern.search(text)), None)


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion."""
    p = successes / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return max(0.0, centre - half), (min(1.0, centre + half) if p < 1 else 1.0)


def main():
    jobs = pd.read_parquet(DATA, columns=["company", "department", "role_family"])
    jobs["role_family"] = jobs["role_family"].astype(str)
    jobs["dept_group"] = jobs["department"].map(department_group)
    compatible = {name: ok for name, _, ok in DEPARTMENT_RULES}

    checked = jobs[jobs["dept_group"].notna() & ~jobs["role_family"].isin(EXCLUDED)].copy()
    checked["agrees"] = [fam in compatible[g] for fam, g in zip(checked["role_family"], checked["dept_group"])]
    lo, hi = wilson_interval(int(checked["agrees"].sum()), len(checked))

    by_type = (checked.groupby("role_family")["agrees"].agg(["mean", "size"])
               .rename(columns={"mean": "agreement", "size": "n"}).sort_values("n", ascending=False))
    # Most common disagreements, to show where the title rules and departments differ.
    misses = (checked[~checked["agrees"]].groupby(["role_family", "dept_group"]).size()
              .sort_values(ascending=False).head(8).rename("n").reset_index())

    in_dept = jobs[jobs["dept_group"].notna()]
    result = {
        "method": "Agreement between title-based job type and the employer's department "
                  "(departments mapped to groups by keyword; compatible job types per group)",
        "cross_functional": sorted(CROSS_FUNCTIONAL),
        "n_cross_functional": int(in_dept["role_family"].isin(CROSS_FUNCTIONAL).sum()),
        "n_checked": int(len(checked)), "n_companies": int(checked["company"].nunique()),
        "agreement": float(checked["agrees"].mean()), "ci_low": float(lo), "ci_high": float(hi),
        "by_job_type": by_type.round(3).reset_index().to_dict(orient="records"),
        "top_disagreements": misses.to_dict(orient="records"),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"Agreement {result['agreement']:.1%} (95% CI {lo:.1%}–{hi:.1%}) on {len(checked):,} postings "
          f"from {result['n_companies']:,} companies")
    print(by_type.to_string())
    print(misses.to_string())


if __name__ == "__main__":
    main()
