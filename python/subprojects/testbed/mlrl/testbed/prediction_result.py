"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that store the result of a prediction process.
"""
from dataclasses import dataclass
from typing import Any

from mlrl.testbed.prediction_scope import PredictionScope, PredictionType


@dataclass
class PredictionResult:
    """
    Stores the result of a prediction process.

    Attributes:
        predictions:        A `numpy.ndarray`, `scipy.sparse.spmatrix` or `scipy.sparse.sparray`, shape
                            `(num_examples, num_outputs)`, that stores the predictions for the query examples
        prediction_type:    The type of the predictions
        prediction_scope:   Specifies whether the predictions have been obtained from a global model or incrementally
        predict_time:       The time needed for prediction
    """
    predictions: Any
    prediction_type: PredictionType
    prediction_scope: PredictionScope
    predict_time: float
