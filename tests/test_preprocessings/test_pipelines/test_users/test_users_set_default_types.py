import pandas as pd
from pandas.testing import assert_frame_equal

from brs.preprocessing.pipeline import UsersSetDefaultTypes


def test_users_set_default_types_basic():
    df = pd.DataFrame({
        "User-ID": [1.0, 2.0, 3.0],
        "Location": pd.Series(["NY", pd.NA, "SF"], dtype="object"),
        "Age": pd.Series([25, 30, 22], dtype="float64"),
    })

    step = UsersSetDefaultTypes()
    result = step.transform(df)

    expected = pd.DataFrame({
        "User-ID": pd.Series([1, 2, 3], dtype="int64"),
        "Location": pd.Series(["NY", pd.NA, "SF"], dtype="string"),
        "Age": pd.Series([25, 30, 22], dtype="Float64"),
    })

    assert_frame_equal(result.reset_index(drop=True), expected)


def test_users_set_default_types_fit_transform():
    df = pd.DataFrame({
        "User-ID": [4.0, 5.0],
        "Location": ["TX", "FL"],
        "Age": [40, 35]
    })

    step = UsersSetDefaultTypes()
    result = step.fit_transform(df)

    expected = pd.DataFrame({
        "User-ID": pd.Series([4, 5], dtype="int64"),
        "Location": pd.Series(["TX", "FL"], dtype="string"),
        "Age": pd.Series([40, 35], dtype="Float64"),
    })

    assert_frame_equal(result.reset_index(drop=True), expected)
