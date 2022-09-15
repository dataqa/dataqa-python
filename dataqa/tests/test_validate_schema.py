import numpy as np
import pandas as pd
import pytest

from dataqa.column_mapping import ColumnMapping, PredictionColumn
from dataqa.infer_schema import (
    check_all_columns_in_df,
    check_categorical_columns,
    check_no_duplicate_columns,
    check_numerical_columns,
    check_prediction_columns,
    format_validated_schema,
    validate_schema,
)

DEFAULT_EMPTY_COLUMN_MAPPING_ARGS = [
    ("numerical_columns", []),
    ("categorical_columns", []),
    ("text_columns", []),
    ("time_columns", []),
    ("prediction_columns", []),
]


def create_empty_column_mapping(update_dict: dict):
    column_mapping = dict(DEFAULT_EMPTY_COLUMN_MAPPING_ARGS)
    column_mapping.update(update_dict)
    return ColumnMapping(**column_mapping)


def test_end_to_end_validate_schema(df_and_column_mapping: dict):
    formatted_schema, _ = validate_schema(
        df_and_column_mapping["df"], df_and_column_mapping["column_mapping"]
    )
    assert formatted_schema == df_and_column_mapping["expected_formatted_schema"]


def test_check_no_duplicate_columns():

    duplicate_numerical = create_empty_column_mapping(
        {"numerical_columns": ["a", "b", "a"]}
    )

    duplicate_across = create_empty_column_mapping(
        {"numerical_columns": ["a", "b"], "categorical_columns": ["c", "a"]}
    )

    with pytest.raises(Exception, match=r"Numerical columns have duplicates"):
        check_no_duplicate_columns(duplicate_numerical)

    with pytest.raises(Exception, match=r"Columns appear across multiple types"):
        check_no_duplicate_columns(duplicate_across)

    no_duplicates = create_empty_column_mapping(
        {
            "prediction_columns": [
                PredictionColumn(
                    prediction_column="a",
                    ground_truth_column="b",
                    task="regression",
                ),
                PredictionColumn(
                    prediction_column="c",
                    ground_truth_column="b",
                    task="regression",
                ),
            ]
        }
    )
    check_no_duplicate_columns(no_duplicates)

    prediction_duplicates = create_empty_column_mapping(
        {
            "prediction_columns": [
                PredictionColumn(
                    prediction_column="a",
                    ground_truth_column="b",
                    task="regression",
                ),
                PredictionColumn(
                    prediction_column="a",
                    ground_truth_column="c",
                    task="regression",
                ),
            ]
        }
    )

    with pytest.raises(Exception, match=r"Prediction columns have duplicates"):
        check_no_duplicate_columns(prediction_duplicates)


def test_check_all_columns_in_df():
    df = pd.DataFrame([], columns=["a", "b", "c"])
    missing_column = create_empty_column_mapping({"numerical_columns": ["a", "d"]})

    with pytest.raises(Exception, match=r"Column d missing from dataframe"):
        check_all_columns_in_df(df, missing_column)

    missing_prediction_column = create_empty_column_mapping(
        {
            "prediction_columns": [
                PredictionColumn(
                    prediction_column="a",
                    ground_truth_column="d",
                    task="regression",
                ),
            ]
        }
    )

    with pytest.raises(Exception, match=r"Column d missing from dataframe"):
        check_all_columns_in_df(df, missing_prediction_column)


def test_check_categorical_columns():
    col1 = pd.Series(["1", "4", "9", "20", "8"], dtype="string", name="col1")
    col2 = pd.Series([1, 2, 3, 4, 5], name="col2")
    col3 = pd.Series(["1", "4", "9", "20", "8"], dtype="category", name="col3")
    col4 = pd.Series([True, False, False, True], name="col4")

    df = pd.concat([col1, col2, col3, col4], axis=1)

    check_categorical_columns(df, ["col1", "col2", "col3", "col4"])

    col1 = pd.Series([f"{x}" for x in range(21)], dtype="string", name="col1")
    with pytest.raises(
        Exception, match=r"Categorical column col1 cannot have more than 20 categories"
    ):
        check_categorical_columns(col1.to_frame(), ["col1"])

    col1 = pd.Series([1, 2, 3, 4, 5, "b"], name="col1")
    with pytest.raises(
        Exception, match=r"Categorical column col1 cannot be of type mixed-integer"
    ):
        check_categorical_columns(col1.to_frame(), ["col1"])


def test_check_numerical_columns():
    col1 = pd.Series([1.0, -20, np.nan, None], name="col1")
    col2 = pd.Series([1, 2, 3, 4, 5], name="col2")
    df = pd.concat([col1, col2], axis=1)

    check_numerical_columns(df, ["col1", "col2"])

    col1 = pd.Series([1, 2, 3, 4, 5], name="col1", dtype="string")
    with pytest.raises(Exception, match=r"Column col1 is not of type numerical"):
        check_numerical_columns(col1.to_frame(), ["col1"])


def test_check_prediction_columns_missing_type():
    missing_type = create_empty_column_mapping(
        {
            "numerical_columns": ["a", "b", "c"],
            "prediction_columns": [
                PredictionColumn(
                    prediction_column="a",
                    ground_truth_column="d",
                    task="regression",
                ),
            ],
        }
    )

    with pytest.raises(
        Exception, match=r"Ground-truth column d not defined as a column type"
    ):
        check_prediction_columns(missing_type, {})


def test_check_prediction_columns_same_column():
    same_column = create_empty_column_mapping(
        {
            "numerical_columns": ["a", "b", "c"],
            "prediction_columns": [
                PredictionColumn(
                    prediction_column="a",
                    ground_truth_column="a",
                    task="regression",
                ),
            ],
        }
    )

    with pytest.raises(
        Exception, match=r"Prediction and ground truth columns cannot be the same"
    ):
        check_prediction_columns(same_column, {})


def test_check_prediction_columns_different_type():
    different_type = create_empty_column_mapping(
        {
            "numerical_columns": ["a", "b", "c"],
            "categorical_columns": ["d"],
            "prediction_columns": [
                PredictionColumn(
                    prediction_column="a",
                    ground_truth_column="d",
                    task="regression",
                ),
            ],
        }
    )

    with pytest.raises(
        Exception,
        match=r"Prediction and ground-truth columns cannot be different types",
    ):
        check_prediction_columns(different_type, {})


def test_check_prediction_columns_categorical_subset():
    different_categories = create_empty_column_mapping(
        {
            "categorical_columns": ["a", "b"],
            "prediction_columns": [
                PredictionColumn(
                    prediction_column="a",
                    ground_truth_column="b",
                    task="classification",
                ),
            ],
        }
    )

    with pytest.raises(
        Exception,
        match=r"Categories of prediction column a are not a subset of categories of ground-truth column b",
    ):
        check_prediction_columns(different_categories, {"a": [1, 2, 3], "b": [1, 2, 4]})


def test_check_prediction_columns_classification():
    classification_mapping = create_empty_column_mapping(
        {
            "text_columns": ["a", "b"],
            "prediction_columns": [
                PredictionColumn(
                    prediction_column="a",
                    ground_truth_column="b",
                    task="classification",
                ),
            ],
        }
    )

    with pytest.raises(
        Exception,
        match=r"Classification tasks only valid with categorical or numerical columns",
    ):
        check_prediction_columns(classification_mapping, {})


def test_check_prediction_columns_regression():
    regression_mapping = create_empty_column_mapping(
        {
            "categorical_columns": ["a", "b"],
            "prediction_columns": [
                PredictionColumn(
                    prediction_column="a",
                    ground_truth_column="b",
                    task="regression",
                ),
            ],
        }
    )

    with pytest.raises(
        Exception,
        match=r"Regression tasks only valid with numerical columns",
    ):
        check_prediction_columns(regression_mapping, {"a": [1, 2], "b": [1, 2, 3]})


def test_check_prediction_columns_bad_task():
    bad_task = create_empty_column_mapping(
        {
            "categorical_columns": ["a", "b"],
            "prediction_columns": [
                PredictionColumn(
                    prediction_column="a",
                    ground_truth_column="b",
                    task="whatever",
                ),
            ],
        }
    )

    with pytest.raises(
        Exception,
        match=r"Prediction task whatever is not supported.",
    ):
        check_prediction_columns(bad_task, {"a": [1, 2], "b": [1, 2, 3]})


def test_format_validated_schema():
    df = pd.DataFrame([], columns=["a", "b", "c"])
    schema_dict = {
        "a": {"type": "categorical"},
        "b": {"type": "text"},
        "c": {"type": "categorical"},
    }
    prediction_columns = [
        PredictionColumn(
            prediction_column="a", ground_truth_column="c", task="classification"
        )
    ]
    column_to_categories = {"a": [1, 2, 3], "c": [1, 2, 3, 4]}

    formatted_schema = format_validated_schema(
        df, schema_dict, prediction_columns, column_to_categories
    )

    assert formatted_schema == [
        {
            "column": "a",
            "type": "categorical",
            "categories": [1, 2, 3],
            "prediction": True,
            "ground_truth": "c",
            "prediction_task": "classification",
        },
        {"column": "b", "type": "text", "prediction": False},
        {
            "column": "c",
            "type": "categorical",
            "categories": [1, 2, 3, 4],
            "prediction": False,
        },
    ]
