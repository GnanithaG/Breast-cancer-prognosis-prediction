import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.harmonize import harmonize_registry_codes  # noqa: E402
from src.data.ingest import load_csv  # noqa: E402


@pytest.fixture(scope="session")
def raw_csv():
    return ROOT / "data" / "Breast_Cancer.csv"


@pytest.fixture(scope="session")
def clean_df(raw_csv):
    return harmonize_registry_codes(load_csv(raw_csv))
