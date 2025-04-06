"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for printing the predictions of a model. The predictions can be written to one or several outputs,
e.g., to the console or to a file.
"""
from dataclasses import replace
from typing import List, Optional

import numpy as np

from mlrl.common.config.options import Options

from mlrl.testbed.data import ArffMetaData, save_arff_file
from mlrl.testbed.dataset import Attribute, AttributeType, Dataset
from mlrl.testbed.experiments.output.data import OutputData
from mlrl.testbed.experiments.output.predictions.predictions import Predictions
from mlrl.testbed.experiments.output.sinks import Sink
from mlrl.testbed.experiments.output.sinks.sink import DatasetFileSink
from mlrl.testbed.experiments.output.writer import DataExtractor, OutputWriter
from mlrl.testbed.experiments.problem_type import ProblemType
from mlrl.testbed.experiments.state import ExperimentState
from mlrl.testbed.util.format import OPTION_DECIMALS
from mlrl.testbed.util.io import SUFFIX_ARFF


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
                prediction_dataset = replace(dataset, y=prediction_result.predictions)
                return Predictions(original_dataset=dataset, prediction_dataset=prediction_dataset)

            return None

    class ArffFileSink(DatasetFileSink):
        """
        Allows to write datasets to an ARFF file.
        """

        def __init__(self, directory: str, options: Options = Options()):
            """
            :param directory:   The path to the directory, where the ARFF file should be located
            :param options:     Options to be taken into account
            """
            super().__init__(directory=directory, suffix=SUFFIX_ARFF, options=options)

        def _write_dataset_to_file(self, file_path: str, state: ExperimentState, dataset: Dataset, **_):
            decimals = self.options.get_int(OPTION_DECIMALS, 0)
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

                if decimals > 0:
                    predictions = np.around(predictions, decimals=decimals)

            features = []
            outputs = []

            for output in dataset.outputs:
                features.append(Attribute('Ground Truth ' + output.name, attribute_type, nominal_values))
                outputs.append(Attribute('Prediction ' + output.name, attribute_type, nominal_values))

            save_arff_file(file_path, dataset.x, predictions, ArffMetaData(features, outputs))

    def __init__(self, *extractors: DataExtractor):
        """
        :param extractors: Extractors that should be used for extracting the output data to be written to the sinks
        """
        super().__init__(*extractors, PredictionWriter.DefaultExtractor())
