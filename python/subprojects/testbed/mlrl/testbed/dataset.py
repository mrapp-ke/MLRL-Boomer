"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for representing datasets.
"""
from dataclasses import dataclass
from enum import Enum
from functools import reduce
from typing import List

from scipy.sparse import lil_array

from mlrl.testbed.data import Attribute, AttributeType


@dataclass
class MetaData:
    """
    The meta-data of a data set.

    Attributes:
        features:           A list that contains all features in the data set
        outputs:            A list that contains all outputs in the data set
    """
    features: List[Attribute]
    outputs: List[Attribute]

    def get_num_features(self, *feature_types: AttributeType) -> int:
        """
        Returns the number of features with one out of a given set of types.  If no types are given, all features are
        counted.

        :param feature_types:   The types of the features to be counted
        :return:                The number of features of the given types
        """
        feature_types = set(feature_types)

        if feature_types:
            return reduce(lambda aggr, feature: aggr + (1 if feature.attribute_type in feature_types else 0),
                          self.features, 0)

        return len(self.features)

    def get_feature_indices(self, *feature_types: AttributeType) -> List[int]:
        """
        Returns a list that contains the indices of all features with one out of a given set of types (in ascending
        order). If no types are given, all indices are returned.

        :param feature_types:   The types of the features whose indices should be returned
        :return:                A list that contains the indices of all features of the given types
        """
        feature_types = set(feature_types)
        return [
            i for i, feature in enumerate(self.features) if not feature_types or feature.attribute_type in feature_types
        ]


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
