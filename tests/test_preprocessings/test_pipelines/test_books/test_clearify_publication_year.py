import pandas as pd
from pandas.testing import assert_frame_equal

from brs.preprocessing.pipeline.steps.books.clearify_publication_year import ClearifyPublicationYear
import brs.config as config


def test_clearify_publication_year_basic():
    df = pd.DataFrame({
        "ISBN": ["111", "222", "333", "444"],
        "Year-Of-Publication": [0, 2000, config.dataset_current_year + 1, 2020],
    })

    step = ClearifyPublicationYear()
    result = step.transform(df)

    expected = pd.DataFrame({
        "ISBN": ["111", "222", "333", "444"],
        "Year-Of-Publication": pd.Series([pd.NA, 2000, pd.NA, 2020], dtype="Int64"),
    })

    assert_frame_equal(result.reset_index(drop=True), expected)


def test_clearify_publication_year_fit_transform_equivalence():
    df = pd.DataFrame({
        "ISBN": ["555", "666"],
        "Year-Of-Publication": [0, config.dataset_current_year + 10],
    })

    step = ClearifyPublicationYear()
    result1 = step.fit_transform(df)
    result2 = step.transform(df)

    expected = pd.DataFrame({
        "ISBN": ["555", "666"],
        "Year-Of-Publication": pd.Series([pd.NA, pd.NA], dtype="Int64"),
    })

    assert_frame_equal(result1.reset_index(drop=True), expected)
    assert_frame_equal(result2.reset_index(drop=True), expected)
