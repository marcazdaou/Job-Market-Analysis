"""Unit tests for the statistical helpers in analysis/."""
import numpy as np
import pytest

from label_validation import department_group, wilson_interval
from salary_statistics import weighted_median


def test_weighted_median_with_equal_weights_is_the_median():
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert weighted_median(values, np.ones(5, int)) == 3.0


def test_weighted_median_respects_weights():
    values = np.array([1.0, 2.0, 3.0])
    assert weighted_median(values, np.array([0, 0, 5])) == 3.0   # all weight on the last value
    assert weighted_median(values, np.array([5, 1, 1])) == 1.0


@pytest.mark.parametrize("successes, n", [(270, 300), (300, 300), (0, 50), (25, 50)])
def test_wilson_interval_is_valid(successes, n):
    lo, hi = wilson_interval(successes, n)
    assert 0 <= lo <= successes / n <= hi <= 1


@pytest.mark.parametrize("department, expected", [
    ("Engineering", "Engineering & technology"),
    ("Business Development", "Sales"),          # not "Engineering" despite "development"
    ("People", "People & HR"),
    ("Product Management", "Product & design"),
    ("Data", "Data"),
    ("Operations", None),                       # ambiguous names are not used
    (None, None),
])
def test_department_group(department, expected):
    assert department_group(department) == expected


def test_wilson_interval_known_value():
    lo, hi = wilson_interval(270, 300)          # 90% of 300
    assert lo == pytest.approx(0.861, abs=0.002)
    assert hi == pytest.approx(0.929, abs=0.002)
