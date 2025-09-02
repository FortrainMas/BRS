import pandas as pd
from pandas.testing import assert_frame_equal

from brs.preprocessing.pipeline import RemoveDropouts


def test_remove_dropouts_default_thresholds():
    df = pd.DataFrame({
        "User-ID": [1, 2, 3, 4],
        "Book-Rating": [1, 2, 3, 4]
    })

    step = RemoveDropouts(min_ratings_threshold=2, max_ratings_threshold=2)
    result = step.transform(df)

    expected = pd.DataFrame(columns=["User-ID", "Book-Rating"])
    expected["User-ID"] = expected["User-ID"].astype("int64")
    expected["Book-Rating"] = expected["Book-Rating"].astype("int64")
    assert_frame_equal(result.reset_index(drop=True), expected)


def test_remove_dropouts_multiple_ratings():
    df = pd.DataFrame({
        "User-ID": [1, 1, 2, 2, 2, 3, 3, 4],
        "Book-Rating": [5, 3, 1, 2, 3, 4, 4, 1]
    })

    step = RemoveDropouts(min_ratings_threshold=2, max_ratings_threshold=2)
    result = step.transform(df)

    expected = pd.DataFrame({
        "User-ID": [1, 1, 3, 3],
        "Book-Rating": [5, 3, 4, 4]
    }).reset_index(drop=True)

    assert_frame_equal(result.reset_index(drop=True), expected)


def test_remove_dropouts_override_with_kwargs():
    df = pd.DataFrame({
        "User-ID": [1, 1, 2, 2, 3, 3],
        "Book-Rating": [5, 3, 1, 2, 4, 4]
    })

    step = RemoveDropouts(min_ratings_threshold=10, max_ratings_threshold=100)
    result = step.transform(df, min_ratings_threshold=1, max_ratings_threshold=2)

    expected = pd.DataFrame({
        "User-ID": [1, 1, 2, 2, 3, 3],
        "Book-Rating": [5, 3, 1, 2, 4, 4]
    }).reset_index(drop=True)

    assert_frame_equal(result.reset_index(drop=True), expected)


def test_fit_transform_equivalence():
    df = pd.DataFrame({
        "User-ID": [1, 1, 2, 2, 3],
        "Book-Rating": [1, 2, 3, 4, 5]
    })

    step = RemoveDropouts(min_ratings_threshold=2, max_ratings_threshold=3)
    result1 = step.fit_transform(df)
    result2 = step.transform(df)

    expected = pd.DataFrame({
        "User-ID": [1, 1, 2, 2],
        "Book-Rating": [1, 2, 3, 4]
    }).reset_index(drop=True)

    assert_frame_equal(result1.reset_index(drop=True), expected)
    assert_frame_equal(result2.reset_index(drop=True), expected)
