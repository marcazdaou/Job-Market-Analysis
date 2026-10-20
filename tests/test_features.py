"""Unit tests for the shared rules in pipeline/features.py."""
import pandas as pd
import pytest

from features import OTHER_FAMILY, clean_annual_usd, model_frame, role_family, seniority


@pytest.mark.parametrize("title, expected", [
    ("Senior Software Engineer", "Software Engineering"),
    ("Machine Learning Engineer", "Data & AI"),        # specific family wins over generic "engineer"
    ("Data Engineer", "Data & AI"),
    ("Product Manager, Payments", "Product & Design"),
    ("Mechanical Engineer", "Hardware & Engineering"),
    ("Account Executive", "Sales & Business Dev."),
    ("Registered Nurse", "Healthcare"),
    ("Barista", "Retail & Hospitality"),
    ("Ingénieur Logiciel H/F", "Software Engineering"),  # French: accents must survive normalization
    ("Verkäufer (m/w/d)", "Retail & Hospitality"),       # German
    ("Net Revenue Manager", "Management"),               # "net" must not match .NET
    ("Senior Sales Engineer", "Sales & Business Dev."),  # an "engineer" title that is a sales job
    ("Technical Support Engineer", "Customer Success & Support"),
    ("Forward Deployed Engineer", "Software Engineering"),
    ("Mechanical Design Engineer", "Hardware & Engineering"),
    ("Innovation Specialist", OTHER_FAMILY),
])
def test_role_family(title, expected):
    assert role_family(title) == expected


@pytest.mark.parametrize("title, expected", [
    ("Junior Data Analyst", "Entry / Junior"),
    ("Software Engineer", "Mid / Unspecified"),
    ("Senior Software Engineer", "Senior / Lead"),
    ("Staff Engineer", "Principal / Staff"),
    ("Staff Nurse", "Mid / Unspecified"),               # "staff nurse" is not a seniority level
    ("Product Manager", "Mid / Unspecified"),           # "manager" alone is not seniority
    ("VP of Engineering", "Director / Executive"),
])
def test_seniority(title, expected):
    assert seniority(title) == expected


def test_missing_title_is_handled():
    assert role_family(None) == OTHER_FAMILY
    assert seniority(None) == "Mid / Unspecified"


def test_clean_annual_usd_assigns_one_status_per_rule():
    raw = pd.DataFrame({
        "lo":  [100_000, None, 50_000, 0,       10,  50_000, 2_000_000],
        "hi":  [140_000, None, 60_000, 50_000,  20,  400_000, 3_000_000],
        "cur": ["USD",   "USD", "EUR", "USD",   "USD", "USD", "USD"],
        "per": ["year",  "year", "year", "year", "year", "year", "year"],
    })
    mid, status = clean_annual_usd(raw["lo"], raw["hi"], raw["cur"], raw["per"])
    assert status.tolist() == ["usable", "no_salary", "not_annual_usd", "invalid_range",
                               "implausible_amount", "range_too_wide", "implausible_amount"]
    assert mid.iloc[0] == 120_000            # midpoint of the range
    assert mid.iloc[1:].isna().all()         # every excluded row has no salary


def test_model_frame_has_expected_columns_and_no_missing_values():
    row = pd.DataFrame([{"title": "Senior Data Engineer", "company": None, "industry": None, "department": None,
                         "country": "United States", "city": "New York", "employment_type": "Full Time",
                         "remote": True, "skills": ["python", "sql"]}])
    frame = model_frame(row)
    assert set(frame.columns) == {"title", "company", "industry", "department", "country", "city",
                                  "employment_type", "seniority", "role_family", "remote", "skills"}
    assert not frame.isna().any().any()
    assert frame.loc[0, "skills"] == "python sql"
    assert frame.loc[0, "remote"] == "Remote"
