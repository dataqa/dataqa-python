from typing import Optional

from dataclasses import dataclass


class PredictionTask:
    # Anything else is not supported
    REGRESSION: str = "regression"
    CLASSIFICATION: str = "classification"


ALL_PREDICTION_TASKS = [PredictionTask.REGRESSION, PredictionTask.CLASSIFICATION]


@dataclass
class PredictionColumn:
    prediction_column: str
    ground_truth_column: str
    task: str


@dataclass
class ColumnMapping:
    numerical_columns: Optional[list[str]] = None
    categorical_columns: Optional[list[str]] = None
    text_columns: Optional[list[str]] = None
    time_columns: Optional[list[str]] = None
    prediction_columns: Optional[list[PredictionColumn]] = None


class ColumnType:
    # Anything else is not supported
    CATEGORICAL: str = "categorical"
    NUMERICAL: str = "numerical"
    TEXT: str = "text"
    TIME: str = "time"
