"""
Author Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow writing datasets to ARFF files.
"""
import numpy as np

from mlrl.common.config.options import Options

from mlrl.testbed.data import ArffMetaData, save_arff_file
from mlrl.testbed.dataset import Attribute, AttributeType, Dataset
from mlrl.testbed.experiments.output.sinks.sink import DatasetFileSink
from mlrl.testbed.experiments.problem_type import ProblemType
from mlrl.testbed.experiments.state import ExperimentState
from mlrl.testbed.util.format import OPTION_DECIMALS


class ArffFileSink(DatasetFileSink):
    """
    Allows to write a dataset to an ARFF file.
    """

    SUFFIX_ARFF = 'arff'

    def __init__(self, directory: str, options: Options = Options()):
        """
        :param directory:   The path to the directory, where the ARFF file should be located
        :param options:     Options to be taken into account
        """
        super().__init__(directory=directory, suffix=self.SUFFIX_ARFF, options=options)

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
