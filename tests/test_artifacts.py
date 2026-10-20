"""Integrity checks on the committed data, model and analysis results.

These protect the live dashboard: if a rebuild produces a broken or
inconsistent file, the tests fail before it is pushed.
"""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from features import SALARY_CEILING, SALARY_FLOOR, model_frame

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def jobs():
    return pd.read_parquet(ROOT / "data" / "jobs_clean.parquet")


@pytest.fixture(scope="module")
def meta():
    return json.loads((ROOT / "models" / "model_meta.json").read_text(encoding="utf-8"))


def test_data_matches_summary(jobs):
    summary = json.loads((ROOT / "data" / "data_summary.json").read_text(encoding="utf-8"))
    assert len(jobs) == summary["postings"]
    assert jobs["salary_mid"].notna().sum() == summary["usable_salaries"]
    assert sum(summary["salary_status_counts"].values()) == summary["postings"]


def test_salaries_are_within_rules(jobs):
    sal = jobs.dropna(subset=["salary_mid"])
    assert sal["salary_mid"].between(SALARY_FLOOR, SALARY_CEILING).all()
    assert (sal["salary_max"] >= sal["salary_min"]).all()
    assert np.allclose(sal["salary_mid"], (sal["salary_min"] + sal["salary_max"]) / 2)
    # A salary is present exactly when the status says it is usable.
    assert (jobs["salary_mid"].notna() == jobs["salary_status"].eq("usable")).all()


def test_urls_are_unique(jobs):
    assert jobs["url"].is_unique


def test_model_loads_and_predicts_plausible_salaries(jobs):
    with (ROOT / "models" / "salary_model.pkl").open("rb") as f:
        model = pickle.load(f)["model"]
    sample = jobs.dropna(subset=["salary_mid"]).sample(500, random_state=0)
    predictions = np.exp(model.predict(model_frame(sample)))
    assert predictions.min() > 10_000 and predictions.max() < 2_000_000
    # On rows it was trained on, the typical error should be small.
    assert np.median(np.abs(predictions / sample["salary_mid"] - 1)) < 0.2


def test_model_metrics_are_consistent(meta):
    test = meta["test"]
    assert test["r2"] > 0.7                                   # guards against a silent regression
    assert 0.75 <= test["interval_coverage"] <= 0.85         # the 80% range is calibrated
    assert meta["unseen_company_test"]["r2"] < test["r2"]    # the stricter test is harder
    best = min(meta["comparison"], key=lambda m: meta["comparison"][m]["cv_mae"])
    assert best == meta["model"]                              # selection rule was followed


def test_confidence_intervals_contain_their_estimates():
    stats = json.loads((ROOT / "analysis" / "results" / "statistics.json").read_text(encoding="utf-8"))
    medians = stats["medians"]
    groups = [medians["overall"]] + [v for k in ["role_family", "seniority", "remote"]
                                     for v in medians[k].values()]
    for g in groups:
        assert g["ci_low"] <= g["median"] <= g["ci_high"]
    for r in stats["remote_regressions"]:
        assert r["ci_low"] <= r["effect"] <= r["ci_high"]


def test_label_validation_is_consistent():
    check = json.loads((ROOT / "analysis" / "results" / "label_validation.json").read_text(encoding="utf-8"))
    assert check["ci_low"] <= check["agreement"] <= check["ci_high"]
    assert check["agreement"] > 0.8               # guards against a rule change that breaks job types
    assert sum(r["n"] for r in check["by_job_type"]) == check["n_checked"]
