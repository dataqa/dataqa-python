import pandas as pd
import pytest

from dataqa.column_mapping import ColumnMapping


@pytest.fixture
def df_and_column_mapping():
    col1 = pd.Series(["1", "1", "9", "20", "8"], dtype="string", name="cat_string")
    col2 = pd.Series([1, 5, 9, 20, 8], name="cat_num")
    col3 = pd.Series([1, 2, 3, 4, 5], name="num")
    col4 = pd.Series(["1", "4", "9", "20", "8"], dtype="string", name="text")
    col5 = pd.Series(
        ["3/11/2000", "3/12/2000", "3/13/2000", "3/13/2022", "3/13/2019"], name="date"
    )

    df = pd.concat([col1, col2, col3, col4, col5], axis=1)

    column_mapping = ColumnMapping(
        categorical_columns=["cat_string", "cat_num"],
        numerical_columns=["num"],
        text_columns=["text"],
        time_columns=["date"],
    )

    expected_formatted_schema = [
        {
            "column": "cat_string",
            "type": "categorical",
            "categories": ["1", "9", "20", "8"],
            "prediction": False,
        },
        {
            "column": "cat_num",
            "type": "categorical",
            "categories": [1, 5, 9, 20, 8],
            "prediction": False,
        },
        {"column": "num", "type": "numerical", "prediction": False},
        {"column": "text", "type": "text", "prediction": False},
        {"column": "date", "type": "time", "prediction": False},
    ]
    return {
        "df": df,
        "column_mapping": column_mapping,
        "expected_formatted_schema": expected_formatted_schema,
    }
