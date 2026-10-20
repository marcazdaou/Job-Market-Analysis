"""Smoke test: the dashboard runs end to end without errors.

Uses Streamlit's built-in test runner, which executes the script headlessly.
"""
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "dashboard" / "app.py")


def test_dashboard_runs_and_predicts():
    app = AppTest.from_file(APP, default_timeout=300).run()
    assert not app.exception

    # Submit the salary estimator form with its default inputs.
    submit = next(b for b in app.button if b.label == "Estimate salary")
    submit.click().run()
    assert not app.exception
    assert any("Estimated salary per year" in m.value for m in app.markdown)


def test_dashboard_handles_filters():
    app = AppTest.from_file(APP, default_timeout=300).run()
    app.sidebar.multiselect[0].select("Data & AI").run()
    app.sidebar.selectbox[0].select("United States").run()
    assert not app.exception
