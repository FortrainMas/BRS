import pandas as pd
from pandas.testing import assert_frame_equal

from brs.preprocessing.pipeline.steps.books.set_default_types import SetDefaultTypes


def test_set_default_types_basic():
    df = pd.DataFrame({
        "ISBN": ["123", "456", "789"],
        "Book-Title": ["A", "B", "C"],
        "Book-Author": ["X", "Y", "Z"],
        "Publisher": ["P1", "P2", "P3"],
        "Year-Of-Publication": ["2000", "abcd", "2010"],
        "Image-URL-S": ["s1", "s2", "s3"],
        "Image-URL-M": ["m1", "m2", "m3"],
        "Image-URL-L": ["l1", "l2", "l3"],
    })

    step = SetDefaultTypes()
    result = step.transform(df)

    expected = pd.DataFrame({
        "ISBN": pd.Series(["123", "789"], dtype="string"),
        "Book-Title": pd.Series(["A", "C"], dtype="string"),
        "Book-Author": pd.Series(["X", "Z"], dtype="string"),
        "Publisher": pd.Series(["P1", "P3"], dtype="string"),
        "Year-Of-Publication": pd.Series([2000, 2010], dtype="Int64"),
    })

    assert_frame_equal(result.reset_index(drop=True), expected)


def test_set_default_types_fit_transform_equivalence():
    df = pd.DataFrame({
        "ISBN": ["111", "222"],
        "Book-Title": ["AA", "BB"],
        "Book-Author": ["XX", "YY"],
        "Publisher": ["P1", "P2"],
        "Year-Of-Publication": ["1999", "2001"],
    })

    step = SetDefaultTypes()
    result1 = step.fit_transform(df)
    result2 = step.transform(df)

    expected = pd.DataFrame({
        "ISBN": pd.Series(["111", "222"], dtype="string"),
        "Book-Title": pd.Series(["AA", "BB"], dtype="string"),
        "Book-Author": pd.Series(["XX", "YY"], dtype="string"),
        "Publisher": pd.Series(["P1", "P2"], dtype="string"),
        "Year-Of-Publication": pd.Series([1999, 2001], dtype="Int64"),
    })

    assert_frame_equal(result1.reset_index(drop=True), expected)
    assert_frame_equal(result2.reset_index(drop=True), expected)
