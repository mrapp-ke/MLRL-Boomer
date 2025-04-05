"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for printing the predictions of a model. The predictions can be written to one or several outputs,
e.g., to the console or to a file.
"""
from dataclasses import replace
from typing import Optional

from mlrl.testbed.experiments.output.data import OutputData
from mlrl.testbed.experiments.output.predictions.predictions import Predictions
from mlrl.testbed.experiments.output.writer import OutputWriter
from mlrl.testbed.experiments.state import ExperimentState


class PredictionWriter(OutputWriter):
    """
    Allows to write predictions and the corresponding ground truth to one or several sinks.
    """

    def _generate_output_data(self, state: ExperimentState) -> Optional[OutputData]:
        prediction_result = state.prediction_result

        if prediction_result:
            dataset = state.dataset
            prediction_dataset = replace(dataset, y=prediction_result.predictions)
            return Predictions(original_dataset=dataset, prediction_dataset=prediction_dataset)

        return None
