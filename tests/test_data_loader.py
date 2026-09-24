"""The loader runs against the CSVs committed under data/."""

import pytest

from src.data_loader import load_and_merge_data
from tests.conftest import DATA_DIR

pytestmark = pytest.mark.skipif(
    not (DATA_DIR / "train.csv").exists(), reason="data/ not present"
)

MARKDOWNS = [f"MarkDown{i}" for i in range(1, 6)]


@pytest.fixture(scope="module")
def train_df():
    return load_and_merge_data(str(DATA_DIR))


@pytest.fixture(scope="module")
def test_df():
    return load_and_merge_data(str(DATA_DIR), is_test=True)


def test_merge_keeps_one_row_per_sales_record(train_df):
    # A left join on (Store, Date, IsHoliday) must not fan out rows.
    assert len(train_df) == 421_570


def test_every_row_finds_its_store_and_features(train_df):
    assert (
        train_df[["Type", "Size", "Temperature", "Fuel_Price"]]
        .notna()
        .all()
        .all()
    )


def test_markdowns_filled_for_training(train_df):
    assert train_df[MARKDOWNS].notna().all().all()


def test_test_split_has_no_target(test_df):
    assert "Weekly_Sales" not in test_df.columns
    assert len(test_df) == 115_064
