"""Shared test setup: make the project's script folders importable."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for folder in ["pipeline", "analysis"]:
    sys.path.insert(0, str(ROOT / folder))
