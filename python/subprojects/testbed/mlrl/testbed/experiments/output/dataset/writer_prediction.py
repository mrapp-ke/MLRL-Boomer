"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow writing predictions to one or several sinks.
"""
from dataclasses import replace
from typing import List, Optional

import numpy as np

from mlrl.testbed.experiments.dataset import Attribute, AttributeType
from mlrl.testbed.experiments.output.data import OutputData
from mlrl.testbed.experiments.output.dataset.dataset_prediction import PredictionDataset
from mlrl.testbed.experiments.output.sinks import Sink
from mlrl.testbed.experiments.output.writer import DataExtractor, OutputWriter
from mlrl.testbed.experiments.problem_domain import ClassificationProblem
from mlrl.testbed.experiments.state import ExperimentState


class PredictionWriter(OutputWriter):
    """
    Allows to write predictions to one or several sinks.
    """

    class DefaultExtractor(DataExtractor):
        """
        The extractor to be used by a `PredictionWriter`, by default.
        """

        def extract_data(self, state: ExperimentState, _: List[Sink]) -> Optional[OutputData]:
            """
            See :func:`mlrl.testbed.experiments.output.writer.DataExtractor.extract_data`
            """
            prediction_result = state.prediction_result
            dataset = state.dataset

            if prediction_result and dataset:
                predictions = dataset.y
                nominal_values = None

                if issubclass(predictions.dtype.type, np.integer):
                    if isinstance(state.problem_domain, ClassificationProblem):
                        attribute_type = AttributeType.NOMINAL
                        nominal_values = [str(value) for value in np.unique(predictions)]
                    else:
                        attribute_type = AttributeType.ORDINAL
                else:
                    attribute_type = AttributeType.NUMERICAL

                outputs = dataset.outputs
                outputs = [Attribute('Prediction ' + output.name, attribute_type, nominal_values) for output in outputs]
                return PredictionDataset(replace(dataset, y=predictions, outputs=outputs))

            return None

    def __init__(self, *extractors: DataExtractor, exit_on_error: bool = True):
        """
        :param extractors:      Extractors that should be used for extracting the output data to be written to the sinks
        :param exit_on_error:   True, if the program should exit when an error occurs while writing the output data,
                                False otherwise
        """
        super().__init__(*extractors, PredictionWriter.DefaultExtractor(), exit_on_error=exit_on_error)
