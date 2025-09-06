import pandas as pd
from pandas.testing import assert_frame_equal

from brs.preprocessing.pipeline import RemoveUnrated

def test_remove_unrated_basic():
    df = pd.DataFrame({
        "User-ID": [1, 2, 3, 4],
        "Name": ["Alice", "Bob", "Charlie", "David"]
    })

    ratings_df = pd.DataFrame({
        "User-ID": [2, 4, 4],
        "Book-Rating": [5, 3, 4]
    })

    step = RemoveUnrated(ratings_df=ratings_df)
    result = step.transform(df)

    expected = pd.DataFrame({
        "User-ID": [2, 4],
        "Name": ["Bob", "David"]
    })

    assert_frame_equal(result.reset_index(drop=True), expected)


def test_remove_unrated_override_with_kwargs():
    df = pd.DataFrame({
        "User-ID": [1, 2, 3],
        "Name": ["Alice", "Bob", "Charlie"]
    })

    default_ratings = pd.DataFrame({
        "User-ID": [1, 3],
        "Book-Rating": [4, 5]
    })

    override_ratings = pd.DataFrame({
        "User-ID": [2, 3],
        "Book-Rating": [5, 3]
    })

    step = RemoveUnrated(ratings_df=default_ratings)
    result = step.transform(df, ratings_df=override_ratings)

    expected = pd.DataFrame({
        "User-ID": [2, 3],
        "Name": ["Bob", "Charlie"]
    })

    assert_frame_equal(result.reset_index(drop=True), expected)


def test_remove_unrated_fit_transform_equivalence():
    df = pd.DataFrame({
        "User-ID": [1, 2, 3],
        "Name": ["Alice", "Bob", "Charlie"]
    })

    ratings_df = pd.DataFrame({
        "User-ID": [1, 3, 3]
    })

    step = RemoveUnrated(ratings_df=ratings_df)
    result1 = step.fit_transform(df)
    result2 = step.transform(df)

    expected = pd.DataFrame({
        "User-ID": [1, 3],
        "Name": ["Alice", "Charlie"]
    })

    assert_frame_equal(result1.reset_index(drop=True), expected)
    assert_frame_equal(result2.reset_index(drop=True), expected)
