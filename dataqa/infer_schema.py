from collections import defaultdict
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

from dataqa.column_mapping import (
    ALL_PREDICTION_TASKS,
    ColumnMapping,
    ColumnType,
    PredictionColumn,
    PredictionTask,
)

MAX_CATEGORICAL_UNIQUE = (
    20  # any column with more than 20 unique values is not categorical
)


def is_column_categorical(values: pd.Series) -> bool:
    if values.nunique() <= MAX_CATEGORICAL_UNIQUE:
        return True
    return False


def infer_schema(
    df: pd.DataFrame,
    numerical_columns: Optional[list[str]] = None,
    categorical_columns: Optional[list[str]] = None,
    text_columns: Optional[list[str]] = None,
    time_columns: Optional[list[str]] = None,
    prediction_columns: Optional[list[PredictionColumn]] = None,
) -> ColumnMapping:
    """
    The keyword arguments will take precedence over the inferred schema.
    """
    column_mapping = defaultdict(list)

    for column_name, column_type in df.dtypes.items():
        if pd.api.types.is_categorical_dtype(column_type):
            categories = df[column_name].unique().tolist()
            if len(categories) > MAX_CATEGORICAL_UNIQUE:
                raise Exception(
                    f"Categorical variables with more than {MAX_CATEGORICAL_UNIQUE} not supported."
                )
            column_mapping[ColumnType.CATEGORICAL].append(column_name)

        elif pd.api.types.is_bool_dtype(column_type):
            column_mapping[ColumnType.CATEGORICAL].append(column_name)

        elif pd.api.types.is_numeric_dtype(column_type):
            # this could also represent a categorical variable
            column_mapping[ColumnType.NUMERICAL].append(column_name)

        elif pd.api.types.is_datetime64_any_dtype(column_type):
            column_mapping[ColumnType.TIME].append(column_name)

        elif str(column_type) == "string":
            categories = df[column_name].unique().tolist()
            if len(categories) > MAX_CATEGORICAL_UNIQUE:
                column_mapping[ColumnType.TEXT].append(column_name)
            else:
                column_mapping[ColumnType.CATEGORICAL].append(column_name)

        elif str(column_type) == "object":
            # could be mixed type or all strings
            inferred_dtype = pd.api.types.infer_dtype(df[column_name], skipna=True)
            if inferred_dtype == "string":
                if is_column_categorical(df[column_name]):
                    column_mapping[ColumnType.CATEGORICAL].append(column_name)
                else:
                    column_mapping[ColumnType.TEXT].append(column_name)

            else:
                raise Exception(f"Mixed data type {inferred_dtype} is not supported.")

        else:
            raise Exception(f"Data type {column_type} is not supported.")

    if numerical_columns is None:
        numerical_columns = column_mapping[ColumnType.NUMERICAL]

    if categorical_columns is None:
        categorical_columns = column_mapping[ColumnType.CATEGORICAL]

    if text_columns is None:
        text_columns = column_mapping[ColumnType.TEXT]

    if time_columns is None:
        time_columns = column_mapping[ColumnType.TIME]

    return ColumnMapping(
        numerical_columns=numerical_columns,
        categorical_columns=categorical_columns,
        text_columns=text_columns,
        time_columns=time_columns,
        prediction_columns=prediction_columns,
    )


def check_no_duplicate_columns(column_mapping: ColumnMapping):
    prediction_columns = [
        column.prediction_column for column in column_mapping.prediction_columns
    ]

    if len(set(column_mapping.categorical_columns)) < len(
        column_mapping.categorical_columns
    ):
        raise Exception("Categorical columns have duplicates.")

    if len(set(column_mapping.numerical_columns)) < len(
        column_mapping.numerical_columns
    ):
        raise Exception("Numerical columns have duplicates.")

    if len(set(column_mapping.text_columns)) < len(column_mapping.text_columns):
        raise Exception("Text columns have duplicates.")

    if len(set(column_mapping.time_columns)) < len(column_mapping.time_columns):
        raise Exception("Time columns have duplicates.")

    if len(set(prediction_columns)) < len(prediction_columns):
        raise Exception("Prediction columns have duplicates.")

    all_columns = (
        column_mapping.categorical_columns
        + column_mapping.numerical_columns
        + column_mapping.text_columns
        + column_mapping.time_columns
    )

    if len(set(all_columns)) < len(all_columns):
        raise Exception("Columns appear across multiple types.")


def check_all_columns_in_df(df: pd.DataFrame, column_mapping: ColumnMapping):
    prediction_columns = [
        column.prediction_column for column in column_mapping.prediction_columns
    ]
    prediction_columns += [
        column.ground_truth_column for column in column_mapping.prediction_columns
    ]

    all_columns = (
        column_mapping.categorical_columns
        + column_mapping.numerical_columns
        + column_mapping.text_columns
        + column_mapping.time_columns
        + prediction_columns
    )
    df_columns = np.array(df.columns)

    for column in all_columns:
        indices = np.where(df_columns == column)[0]
        if len(indices) > 1:
            raise Exception("Duplicated column names in dataframe.")
        if len(indices) == 0:
            raise Exception(f"Column {column} missing from dataframe.")


def check_categorical_columns(
    df: pd.DataFrame, categorical_columns: list[str]
) -> dict[str, list[Union[str, np.number]]]:
    """
    Make sure the dtype is numeric or string (not mixed) and that unique categories <= MAX_CATEGORICAL_UNIQUE
    """
    column_to_categories = {}
    for column in categorical_columns:
        inferred_dtype = pd.api.types.infer_dtype(df[column], skipna=True)
        if not (
            inferred_dtype in ["categorical", "boolean", "string"]
            or pd.api.types.is_numeric_dtype(df[column].dtype)
        ):
            raise Exception(
                f"Categorical column {column} cannot be of type {inferred_dtype}. "
                "It needs to be string, numeric, boolean or categorical."
            )
        categories = df[column].unique().tolist()
        if len(categories) > MAX_CATEGORICAL_UNIQUE:
            raise Exception(
                f"Categorical column {column} cannot have more than {MAX_CATEGORICAL_UNIQUE} categories."
            )
        column_to_categories[column] = categories
    return column_to_categories


def check_numerical_columns(df: pd.DataFrame, numerical_columns: list[str]):
    for column in numerical_columns:
        if not pd.api.types.is_numeric_dtype(df[column].dtype):
            raise Exception(f"Column {column} is not of type numerical.")


def check_text_columns(df: pd.DataFrame, text_columns: list[str]):
    for column in text_columns:
        if not pd.api.types.infer_dtype(df[column], skipna=True) == "string":
            raise Exception(f"Text column {column} is not of type string.")


def check_time_columns(df: pd.DataFrame, time_columns: list[str]):
    for column in time_columns:
        try:
            _ = pd.to_datetime(df[column], errors="raise")
        except:
            raise Exception(f"Column {column} cannot be cast to a datetime.")


def is_subset(list1: list[Any], list2: list[Any]) -> bool:
    return len(set(list1).difference(set(list2))) == 0


def check_prediction_columns(
    column_mapping: ColumnMapping,
    column_to_categories: dict[str, list[Union[str, np.number]]],
) -> dict:
    schema_dict = dict(
        (column, {"type": ColumnType.CATEGORICAL})
        for column in column_mapping.categorical_columns
    )
    schema_dict.update(
        dict(
            (column, {"type": ColumnType.NUMERICAL})
            for column in column_mapping.numerical_columns
        )
    )
    schema_dict.update(
        dict(
            (column, {"type": ColumnType.TEXT})
            for column in column_mapping.text_columns
        )
    )
    schema_dict.update(
        dict(
            (column, {"type": ColumnType.TIME})
            for column in column_mapping.time_columns
        )
    )

    for column in column_mapping.prediction_columns:
        prediction_column = column.prediction_column
        ground_truth_column = column.ground_truth_column
        task = column.task

        if prediction_column == ground_truth_column:
            raise Exception("Prediction and ground truth columns cannot be the same.")

        if not prediction_column in schema_dict:
            raise Exception(
                f"Prediction column {prediction_column} not defined as a column type."
            )

        if not ground_truth_column in schema_dict:
            raise Exception(
                f"Ground-truth column {ground_truth_column} not defined as a column type."
            )

        if (
            schema_dict[prediction_column]["type"]
            != schema_dict[ground_truth_column]["type"]
        ):
            raise Exception(
                f"Prediction and ground-truth columns cannot be different types: "
                f"{schema_dict[prediction_column]['type']} "
                f"and {schema_dict[ground_truth_column]['type']}."
            )

        if schema_dict[prediction_column]["type"] == ColumnType.CATEGORICAL:
            if not is_subset(
                column_to_categories[prediction_column],
                column_to_categories[ground_truth_column],
            ):
                raise Exception(
                    f"Categories of prediction column {prediction_column} are not a subset of categories "
                    f"of ground-truth column {ground_truth_column}."
                )

        if task not in ALL_PREDICTION_TASKS:
            raise Exception(
                f"Prediction task {task} is not supported. Only supported tasks: {ALL_PREDICTION_TASKS}."
            )

        if task == PredictionTask.REGRESSION:
            if schema_dict[prediction_column] != ColumnType.NUMERICAL:
                raise Exception(f"Regression tasks only valid with numerical columns.")

        if task == PredictionTask.CLASSIFICATION:
            if not schema_dict[prediction_column]["type"] in [
                ColumnType.CATEGORICAL,
                ColumnType.NUMERICAL,
            ]:
                raise Exception(
                    f"Classification tasks only valid with categorical or numerical columns."
                )

    return schema_dict


def format_validated_schema(
    df: pd.DataFrame,
    schema_dict: dict,
    prediction_columns: list[PredictionColumn],
    column_to_categories: dict[str, list[Union[str, np.number]]],
) -> dict:
    new_schema = []
    prediction_columns_dict = {
        column.prediction_column: column for column in prediction_columns
    }

    for column in df.columns:
        if column in schema_dict:
            column_row = {
                "column": column,
                "type": schema_dict[column]["type"],
                "prediction": False,
            }
            if schema_dict[column]["type"] == ColumnType.CATEGORICAL:
                column_row["categories"] = column_to_categories[column]
            if column in prediction_columns_dict:
                column_row["prediction"] = True
                column_row["prediction_task"] = prediction_columns_dict[column].task
                column_row["ground_truth"] = prediction_columns_dict[
                    column
                ].ground_truth_column
        new_schema.append(column_row)

    return new_schema


def validate_schema(
    df: pd.DataFrame, column_mapping: ColumnMapping
) -> [ColumnMapping, pd.DataFrame]:
    categorical_columns = column_mapping.categorical_columns or []
    numerical_columns = column_mapping.numerical_columns or []
    text_columns = column_mapping.text_columns or []
    time_columns = column_mapping.time_columns or []
    prediction_columns = column_mapping.prediction_columns or []

    new_column_mapping = ColumnMapping(
        categorical_columns=categorical_columns,
        numerical_columns=numerical_columns,
        text_columns=text_columns,
        time_columns=time_columns,
        prediction_columns=prediction_columns,
    )

    check_no_duplicate_columns(new_column_mapping)
    check_all_columns_in_df(df, new_column_mapping)

    column_to_categories = check_categorical_columns(
        df, new_column_mapping.categorical_columns
    )
    check_numerical_columns(df, new_column_mapping.numerical_columns)
    check_text_columns(df, new_column_mapping.text_columns)
    check_time_columns(df, new_column_mapping.time_columns)

    schema_dict = check_prediction_columns(new_column_mapping, column_to_categories)

    formatted_schema = format_validated_schema(
        df, schema_dict, prediction_columns, column_to_categories
    )

    df = df[[x["column"] for x in formatted_schema]]

    return formatted_schema, df
