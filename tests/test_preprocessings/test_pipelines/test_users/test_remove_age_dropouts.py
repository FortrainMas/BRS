import pandas as pd
from pandas.testing import assert_frame_equal

from brs.preprocessing.pipeline import RemoveAgeDropouts


def test_remove_dropouts_default_thresholds():
    df = pd.DataFrame({
        "User-ID": [1, 2, 3, 4],
        "Age": pd.Series([5.0, 20.0, 150.0, 50.0], dtype="Float64"),
    })

    step = RemoveAgeDropouts(min_age_threshold=10, max_age_threshold=100)
    result = step.transform(df)

    expected = pd.DataFrame({
        "User-ID": [1, 2, 3, 4],
    })
    expected["Age"] = pd.Series([pd.NA, 20.0, pd.NA, 50.0], dtype="Float64")

    assert_frame_equal(result.reset_index(drop=True), expected)

def test_remove_dropouts_override_with_kwargs():
    df = pd.DataFrame({
        "User-ID": [1, 2, 3],
        "Age": [5.0, 20.0, 150.0],
    })

    step = RemoveAgeDropouts(min_age_threshold=10, max_age_threshold=100)
    result = step.transform(df, min_age_threshold=0, max_age_threshold=200)

    expected = df.copy()

    assert_frame_equal(result.reset_index(drop=True), expected)

def test_remove_dropouts_fit_transform():
    df = pd.DataFrame({
        "User-ID": [1, 2, 3, 4],
        "Age": pd.Series([5.0, 20.0, 150.0, 50.0], dtype="Float64"),
    })

    step = RemoveAgeDropouts(min_age_threshold=10, max_age_threshold=100)
    result = step.fit_transform(df)

    expected = pd.DataFrame({
        "User-ID": [1, 2, 3, 4],
    })
    expected["Age"] = pd.Series([pd.NA, 20.0, pd.NA, 50.0], dtype="Float64")

    assert_frame_equal(result.reset_index(drop=True), expected)

def test_remove_dropouts_fit_transform_override_with_kwargs():    
    df = pd.DataFrame({
        "User-ID": [1, 2, 3],
        "Age": [5.0, 20.0, 150.0],
    })

    step = RemoveAgeDropouts(min_age_threshold=10, max_age_threshold=100)
    result = step.fit_transform(df, min_age_threshold=0, max_age_threshold=200)

    expected = df.copy()

    assert_frame_equal(result.reset_index(drop=True), expected)
