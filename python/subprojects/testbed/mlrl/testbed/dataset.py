"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for representing datasets.
"""
from dataclasses import dataclass
from enum import Enum

from scipy.sparse import lil_array

from mlrl.testbed.meta_data import MetaData


@dataclass
class Dataset:
    """
    A dataset consisting of two matrices `x` and `y`, storing the features of examples and their respective ground
    truth, respectively.

    Attributes:
        x:          A `lil_array`, shape `(num_examples, num_features)`, that stores the features of examples
        y:          A `lil_array`, shape `(num_examples, num_features)`, that stores the ground truth of examples
        meta_data:  The meta-data of the dataset
    """
    x: lil_array
    y: lil_array
    meta_data: MetaData

    class Type(Enum):
        """
        Characterizes a dataset as either training or test data.
        """
        TRAINING = 'training'
        TEST = 'test'

        def get_file_name(self, dataset_name: str) -> str:
            """
            Returns the name of a file name that corresponds to a specific type of data.

            :param dataset_name:    The name of the dataset (without suffix)
            :return:                The file name
            """
            return dataset_name + '_' + str(self.value)

    @property
    def num_examples(self) -> int:
        """
        The number of examples in the dataset.
        """
        return self.x.shape[0]

    @property
    def num_features(self) -> int:
        """
        The number of features in the dataset.
        """
        return self.x.shape[1]

    @property
    def num_outputs(self) -> int:
        """
        The number of outputs in the dataset.
        """
        return self.y.shape[1]
