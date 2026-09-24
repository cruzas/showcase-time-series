"""Shared fixtures: small, hand-built frames shaped like the merged Walmart data."""

from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"


def _frame(dates: list[str], seed: int = 0) -> pd.DataFrame:
    rows = []
    for i, date in enumerate(dates):
        for store, store_type, size in [(1, "A", 151315), (2, "B", 93638)]:
            for dept in (1, 7):
                rows.append(
                    {
                        "Store": store,
                        "Dept": dept,
                        "Date": date,
                        "Weekly_Sales": 10_000.0 * store
                        + 500.0 * dept
                        + 37 * i,
                        "IsHoliday": i % 5 == 0,
                        "Type": store_type,
                        "Size": size,
                        "Temperature": 40.0 + i + seed,
                        "Fuel_Price": 2.5 + 0.01 * i,
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture
def train_frame() -> pd.DataFrame:
    return _frame([f"2011-{m:02d}-07" for m in range(1, 13)])


@pytest.fixture
def val_frame() -> pd.DataFrame:
    return _frame([f"2012-{m:02d}-06" for m in range(1, 5)], seed=3)
