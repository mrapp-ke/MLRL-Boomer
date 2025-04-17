"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow writing predictions and the corresponding ground truth to one or several sinks.
"""
from dataclasses import replace
from typing import List, Optional

import numpy as np

from mlrl.testbed.dataset import Attribute, AttributeType
from mlrl.testbed.experiments.output.data import OutputData
from mlrl.testbed.experiments.output.predictions.predictions import Predictions
from mlrl.testbed.experiments.output.sinks import Sink
from mlrl.testbed.experiments.output.writer import DataExtractor, OutputWriter
from mlrl.testbed.experiments.problem_type import ProblemType
from mlrl.testbed.experiments.state import ExperimentState
from mlrl.testbed.util.format import OPTION_DECIMALS


class PredictionWriter(OutputWriter):
    """
    Allows to write predictions and the corresponding ground truth to one or several sinks.
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

            if prediction_result:
                dataset = state.dataset
                predictions = dataset.y
                nominal_values = None

                if issubclass(predictions.dtype.type, np.integer):
                    if state.problem_type == ProblemType.CLASSIFICATION:
                        attribute_type = AttributeType.NOMINAL
                        nominal_values = [str(value) for value in np.unique(predictions)]
                    else:
                        attribute_type = AttributeType.ORDINAL
                else:
                    attribute_type = AttributeType.NUMERICAL
                    decimals = self.options.get_int(OPTION_DECIMALS, 0)

                    if decimals > 0:
                        predictions = np.around(predictions, decimals=decimals)

                outputs = dataset.outputs
                outputs = [Attribute('Prediction ' + output.name, attribute_type, nominal_values) for output in outputs]
                prediction_dataset = replace(dataset, y=predictions, outputs=outputs)
                return Predictions(original_dataset=dataset, prediction_dataset=prediction_dataset)

            return None

    def __init__(self, *extractors: DataExtractor):
        """
        :param extractors: Extractors that should be used for extracting the output data to be written to the sinks
        """
        super().__init__(*extractors, PredictionWriter.DefaultExtractor())
