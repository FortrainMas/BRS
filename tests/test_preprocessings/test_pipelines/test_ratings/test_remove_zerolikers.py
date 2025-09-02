import pandas as pd
from pandas.testing import assert_frame_equal

from brs.preprocessing.pipeline import RemoveZerolikers


def test_remove_zerolikers_basic():
    df = pd.DataFrame({
        "User-ID": [1, 1, 2, 2, 3, 3],
        "Book-Rating": [0, 0, 3, 2, 5, 5]
    })

    step = RemoveZerolikers(zero_likers_threshold=1)
    result = step.transform(df)

    expected = pd.DataFrame({
        "User-ID": [2, 2, 3, 3],
        "Book-Rating": [3, 2, 5, 5]
    }).reset_index(drop=True)

    assert_frame_equal(result.reset_index(drop=True), expected)


def test_remove_zerolikers_with_kwargs_override():
    df = pd.DataFrame({
        "User-ID": [1, 1, 2, 2, 3, 3],
        "Book-Rating": [0, 0, 3, 2, 5, 5]
    })

    step = RemoveZerolikers(zero_likers_threshold=10)
    result = step.transform(df, zero_likers_threshold=1)

    expected = pd.DataFrame({
        "User-ID": [2, 2, 3, 3],
        "Book-Rating": [3, 2, 5, 5]
    }).reset_index(drop=True)

    assert_frame_equal(result.reset_index(drop=True), expected)


def test_fit_transform_equivalence():
    df = pd.DataFrame({
        "User-ID": [1, 2, 3],
        "Book-Rating": [1, 0, 5]
    })

    step = RemoveZerolikers(zero_likers_threshold=0)
    result1 = step.fit_transform(df)
    result2 = step.transform(df)

    expected = pd.DataFrame({
        "User-ID": [1, 3],
        "Book-Rating": [1, 5]
    }).reset_index(drop=True)

    assert_frame_equal(result1.reset_index(drop=True), expected)
    assert_frame_equal(result2.reset_index(drop=True), expected)
