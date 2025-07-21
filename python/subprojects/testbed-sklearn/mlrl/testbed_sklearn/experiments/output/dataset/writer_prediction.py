"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow writing predictions to one or several sinks.
"""

from dataclasses import replace
from typing import List, Optional, override

import numpy as np

from mlrl.testbed_sklearn.experiments.dataset import Attribute, AttributeType, TabularDataset
from mlrl.testbed_sklearn.experiments.output.dataset.dataset_prediction import PredictionDataset

from mlrl.testbed.experiments.output.data import OutputData
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

        @override
        def extract_data(self, state: ExperimentState, _: List[Sink]) -> Optional[OutputData]:
            """
            See :func:`mlrl.testbed.experiments.output.writer.DataExtractor.extract_data`
            """
            prediction_result = state.prediction_result
            dataset = state.dataset_as(self, TabularDataset)

            if prediction_result and dataset:
                predictions = prediction_result.predictions
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

    def __init__(self, *extractors: DataExtractor):
        """
        :param extractors: Extractors that should be used for extracting the output data to be written to the sinks
        """
        super().__init__(*extractors, PredictionWriter.DefaultExtractor())
