"""
Author Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow writing datasets to ARFF files.
"""
from dataclasses import replace

import arff
import numpy as np

from scipy.sparse import dok_array

from mlrl.common.config.options import Options

from mlrl.testbed.dataset import Attribute, AttributeType, Dataset
from mlrl.testbed.experiments.output.sinks.sink import DatasetFileSink
from mlrl.testbed.experiments.problem_type import ProblemType
from mlrl.testbed.experiments.state import ExperimentState
from mlrl.testbed.util.format import OPTION_DECIMALS
from mlrl.testbed.util.io import open_writable_file


class ArffFileSink(DatasetFileSink):
    """
    Allows to write a dataset to an ARFF file.
    """

    SUFFIX_ARFF = 'arff'

    @staticmethod
    def __save_arff_file(file_path: str, dataset: Dataset):
        sparse = dataset.has_sparse_features and dataset.has_sparse_outputs
        num_examples = dataset.num_examples
        num_features = dataset.num_features
        num_outputs = dataset.num_outputs

        features = dataset.features
        x_features = [(features[i].name, 'NUMERIC' if features[i].attribute_type == AttributeType.NUMERICAL
                       or features[i].nominal_values is None else features[i].nominal_values)
                      for i in range(num_features)]

        outputs = dataset.outputs
        y_features = [(outputs[i].name, 'NUMERIC' if outputs[i].attribute_type == AttributeType.NUMERICAL
                       or outputs[i].nominal_values is None else outputs[i].nominal_values) for i in range(num_outputs)]

        if sparse:
            data = [{} for _ in range(num_examples)]
        else:
            data = [[0 for _ in range(num_features + num_outputs)] for _ in range(num_examples)]

        for keys, value in dok_array(dataset.x).items():
            data[keys[0]][keys[1]] = value

        for keys, value in dok_array(dataset.y).items():
            data[keys[0]][num_features + keys[1]] = value

        with open_writable_file(file_path) as arff_file:
            arff_file.write(
                arff.dumps({
                    'description': 'traindata',
                    'relation': 'traindata: -C ' + str(-num_outputs),
                    'attributes': x_features + y_features,
                    'data': data
                }))

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

        self.__save_arff_file(file_path, replace(dataset, y=predictions, features=features, outputs=outputs))
