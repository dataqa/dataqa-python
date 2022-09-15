from datetime import datetime, timedelta

import pandas as pd
import pytest

from dataqa.infer_schema import infer_schema


def test_valid_categorical_column():
    col1 = pd.Series(["1", "4", "9", "20", "8"], dtype="string", name="col1")
    col2 = pd.Series(["1", "4", "9", "20", "8"], name="col2")
    col3 = pd.Series(["1", "4", "9", "20", "8"], dtype="category", name="col3")
    col4 = pd.Series([True, False, False, True], name="col4")

    assert "col1" in infer_schema(col1.to_frame()).categorical_columns
    assert "col2" in infer_schema(col2.to_frame()).categorical_columns
    assert "col3" in infer_schema(col3.to_frame()).categorical_columns
    assert "col4" in infer_schema(col4.to_frame()).categorical_columns


def test_valid_numerical_column():
    col1 = pd.Series([1, 2, 3, 4], name="col1")

    assert "col1" in infer_schema(col1.to_frame()).numerical_columns


def test_valid_text_column():
    col1 = pd.Series([f"{x}" for x in range(21)], dtype="string", name="col1")
    col2 = pd.Series([f"{x}" for x in range(21)], name="col2")

    assert "col1" in infer_schema(col1.to_frame()).text_columns
    assert "col2" in infer_schema(col2.to_frame()).text_columns


def test_valid_time_column():
    col1 = pd.to_datetime(
        pd.Series(["3/11/2000", "3/12/2000", "3/13/2000"], name="col1")
    )
    col2 = pd.Series(
        [datetime(2022, 9, 13, 9), pd.Timestamp(1513393355.5, unit="s")], name="col2"
    )

    assert "col1" in infer_schema(col1.to_frame()).time_columns
    assert "col2" in infer_schema(col2.to_frame()).time_columns


def test_non_supported_mixed_column():
    col1 = pd.Series([1, 2, 3, 4, "a"], name="col1")
    col2 = pd.Series([1, 2, 3, 4, datetime(2013, 1, 1)], name="col1")

    with pytest.raises(
        Exception, match=r"Mixed data type mixed-integer is not supported"
    ):
        infer_schema(col1.to_frame())
        infer_schema(col2.to_frame())


def test_non_supported_other_column():
    col1 = pd.Series(pd.Timedelta(timedelta(days=1, seconds=1)))
    col2 = pd.Series(pd.Interval(left=0, right=5))
    col3 = pd.Series(pd.Period("2017-01-01"))

    with pytest.raises(
        Exception, match=r"Data type timedelta64\[ns\] is not supported"
    ):
        infer_schema(col1.to_frame())

    with pytest.raises(Exception, match=r"Data type interval.* is not supported"):
        infer_schema(col2.to_frame())

    with pytest.raises(Exception, match=r"Data type period.* is not supported"):
        infer_schema(col3.to_frame())
