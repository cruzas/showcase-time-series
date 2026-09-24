"""Feature engineering and leakage-free scaling."""

import numpy as np

from src.preprocessor import DataPreprocessor


def test_time_features(train_frame):
    out = DataPreprocessor().extract_time_features(train_frame)
    first = out.iloc[0]
    assert (first["Year"], first["Month"], first["Week"]) == (2011, 1, 1)
    assert out["IsHoliday"].isin([0, 1]).all()


def test_does_not_mutate_input(train_frame):
    before = train_frame.copy()
    DataPreprocessor().fit_transform(train_frame)
    assert train_frame.equals(before)


def test_fit_transform_shapes_and_scaling(train_frame):
    pre = DataPreprocessor()
    x_cont, x_cat, y = pre.fit_transform(train_frame)
    n = len(train_frame)
    assert x_cont.shape == (n, len(pre.cont_cols))
    assert x_cat.shape == (n, len(pre.cat_cols))
    assert y.shape == (n,)
    np.testing.assert_allclose(y.mean(), 0.0, atol=1e-9)
    np.testing.assert_allclose(y.std(), 1.0, atol=1e-9)


def test_transform_reuses_training_statistics(train_frame, val_frame):
    pre = DataPreprocessor()
    pre.fit_transform(train_frame)
    mean_before = pre.y_scaler.mean_.copy()

    _, _, y_val = pre.transform(val_frame)

    # Fitting on validation data would be leakage; statistics must not move.
    np.testing.assert_array_equal(pre.y_scaler.mean_, mean_before)
    expected = (
        val_frame["Weekly_Sales"] - mean_before[0]
    ) / pre.y_scaler.scale_[0]
    np.testing.assert_allclose(y_val, expected.to_numpy())


def test_target_round_trips_to_dollars(train_frame):
    pre = DataPreprocessor()
    _, _, y = pre.fit_transform(train_frame)
    dollars = pre.y_scaler.inverse_transform(y.reshape(-1, 1)).ravel()
    np.testing.assert_allclose(dollars, train_frame["Weekly_Sales"].to_numpy())
