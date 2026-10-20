"""Statistical inference for the dashboard's headline findings.

Two analyses, both saved to analysis/results/statistics.json for the dashboard,
README and report to read (so the hosted app never has to recompute them):

1. 95% confidence intervals for median salaries (overall, by job type, by
   seniority, remote vs. on-site) using a *company cluster bootstrap*:
   each replicate resamples whole companies with replacement. Postings from
   the same company are not independent (shared pay bands, multi-city copies),
   so resampling individual postings would give intervals that are too narrow.

2. A "fair comparison" of remote vs. on-site pay: OLS regressions of log
   salary on a remote indicator, adding controls step by step (job type,
   seniority, country, employment type, and finally company fixed effects).
   Standard errors are clustered by company.

Run from the project root (after pipeline/collector.py):  python analysis/salary_statistics.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
from features import OTHER_FAMILY  # noqa: E402

DATA = ROOT / "data" / "jobs_clean.parquet"
OUT = ROOT / "analysis" / "results" / "statistics.json"
B = 1000      # bootstrap replicates
SEED = 42


# ── 1. Company cluster bootstrap for medians ──────────────────────────────────
def weighted_median(sorted_values: np.ndarray, weights: np.ndarray) -> float:
    """Median of values (already sorted ascending) where each value has an integer weight."""
    cumulative = np.cumsum(weights)
    return float(sorted_values[np.searchsorted(cumulative, cumulative[-1] / 2)])


def cluster_bootstrap(sal: pd.DataFrame) -> dict:
    """95% percentile intervals for medians, resampling companies with replacement.

    A resampled company that is drawn k times contributes all of its postings
    k times, which is implemented as an integer weight per posting.
    """
    rng = np.random.default_rng(SEED)
    company = pd.factorize(sal["company"])[0]
    n_companies = company.max() + 1

    # Each "cell" is a set of postings whose median we track (overall, each job type, ...).
    cells = {("overall", "All postings"): np.ones(len(sal), bool)}
    for col in ["role_family", "seniority", "remote"]:
        for value in sal[col].dropna().unique():
            if col == "role_family" and value == OTHER_FAMILY:
                continue
            cells[(col, str(value))] = (sal[col] == value).to_numpy()

    # Pre-sort each cell once; per replicate only the weights change.
    prepared = {}
    for key, mask in cells.items():
        values = sal["salary_mid"].to_numpy()[mask]
        order = np.argsort(values)
        prepared[key] = (values[order], company[mask][order])

    draws = {key: np.empty(B) for key in cells}
    for b in range(B):
        counts = np.bincount(rng.integers(0, n_companies, n_companies), minlength=n_companies)
        for key, (values, comp) in prepared.items():
            weights = counts[comp]
            draws[key][b] = weighted_median(values, weights) if weights.sum() else np.nan

    def summarize(key):
        values, _ = prepared[key]
        lo, hi = np.nanpercentile(draws[key], [2.5, 97.5])
        return {"median": float(np.median(values)), "ci_low": float(lo), "ci_high": float(hi),
                "n": int(len(values))}

    out = {"overall": summarize(("overall", "All postings"))}
    for col in ["role_family", "seniority", "remote"]:
        out[col] = {k[1]: summarize(k) for k in cells if k[0] == col}

    # Ratios of medians, with intervals from the same replicates.
    def ratio(a, b):
        r = draws[a] / draws[b]
        lo, hi = np.nanpercentile(r, [2.5, 97.5])
        return {"ratio": float(np.median(prepared[a][0]) / np.median(prepared[b][0])),
                "ci_low": float(lo), "ci_high": float(hi)}

    out["ratios"] = {
        "remote_vs_onsite": ratio(("remote", "True"), ("remote", "False")),
        "principal_vs_entry": ratio(("seniority", "Principal / Staff"), ("seniority", "Entry / Junior")),
    }
    return out


# ── 2. Remote vs. on-site: raw vs. adjusted gap ───────────────────────────────
def remote_regressions(sal: pd.DataFrame) -> list[dict]:
    """Remote coefficient from progressively controlled OLS models of log salary.

    Uses de-duplicated postings (same rule as the ML model) so a job copied to
    many cities is counted once. Each model reports exp(coef) - 1, the % pay
    difference for remote postings, with a company-clustered 95% CI.
    """
    d = sal.drop_duplicates(["company", "title", "salary_min", "salary_max"]).copy()
    d = d[d["role_family"] != OTHER_FAMILY]
    d["log_salary"] = np.log(d["salary_mid"])
    d["remote_i"] = d["remote"].astype(float)
    country = d["country"].astype(str)   # categorical -> plain text so a new label can be added
    top_countries = country.value_counts().index[:10]
    d["country_g"] = country.where(country.isin(top_countries), "Other / unknown")
    d["employment_g"] = d["employment_type"].astype(str).replace("nan", "Unknown")
    groups = pd.factorize(d["company"])[0]

    specs = [
        ("Raw difference", []),
        ("+ job type & seniority", ["role_family", "seniority"]),
        ("+ country & employment type", ["role_family", "seniority", "country_g", "employment_g"]),
        ("+ same company (fixed effects)", ["role_family", "seniority", "country_g", "employment_g"]),
    ]
    results = []
    for label, controls in specs:
        X = pd.get_dummies(d[controls].astype(str), drop_first=True, dtype=float) if controls else \
            pd.DataFrame(index=d.index)
        X.insert(0, "remote", d["remote_i"])
        y = d["log_salary"]
        if label.startswith("+ same company"):
            # Company fixed effects via the within transformation: subtract each
            # company's mean, so only differences *inside* a company remain.
            X = X - X.groupby(groups).transform("mean")
            y = y - y.groupby(groups).transform("mean")
            X = X.loc[:, X.std() > 0]
        else:
            X = sm.add_constant(X)
        fit = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": groups})
        coef, se = fit.params["remote"], fit.bse["remote"]
        results.append({
            "model": label, "n": int(len(d)),
            "effect": float(np.expm1(coef)),
            "ci_low": float(np.expm1(coef - 1.96 * se)), "ci_high": float(np.expm1(coef + 1.96 * se)),
            "p_value": float(fit.pvalues["remote"]),
        })
        print(f"{label:<34} remote effect {np.expm1(coef):+.1%}  "
              f"[{np.expm1(coef - 1.96 * se):+.1%}, {np.expm1(coef + 1.96 * se):+.1%}]  p={fit.pvalues['remote']:.3g}",
              flush=True)
    return results


def main():
    jobs = pd.read_parquet(DATA)
    sal = jobs.dropna(subset=["salary_mid"]).reset_index(drop=True)
    sal["remote"] = sal["remote"].astype(bool)
    sal["role_family"] = sal["role_family"].astype(str)
    sal["seniority"] = sal["seniority"].astype(str)

    print(f"Cluster bootstrap: {len(sal):,} salaried postings, {sal['company'].nunique():,} companies, "
          f"{B} replicates", flush=True)
    medians = cluster_bootstrap(sal)
    o = medians["overall"]
    print(f"Overall median ${o['median']:,.0f}  95% CI [${o['ci_low']:,.0f}, ${o['ci_high']:,.0f}]", flush=True)

    regressions = remote_regressions(sal)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "method": {
            "medians": f"Company cluster bootstrap, {B} replicates, 95% percentile intervals",
            "regression": "OLS of log salary; company-clustered standard errors; de-duplicated postings; "
                          "'Other' job type excluded",
        },
        "medians": medians,
        "remote_regressions": regressions,
    }, indent=1), encoding="utf-8")
    print(f"Saved {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
