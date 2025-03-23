"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for representing datasets.
"""
from dataclasses import dataclass
from functools import reduce
from typing import List

from scipy.sparse import lil_array

from mlrl.testbed.data import Attribute, AttributeType


@dataclass
class Dataset:
    """
    A dataset consisting of two matrices `x` and `y`, storing the features of examples and their respective ground
    truth, respectively.

    Attributes:
        x:          A `lil_array`, shape `(num_examples, num_features)`, that stores the features of examples
        y:          A `lil_array`, shape `(num_examples, num_features)`, that stores the ground truth of examples
        features:   A list that contains the features contained in the dataset
        outputs:    A list that contains the outputs contained in the dataset
    """
    x: lil_array
    y: lil_array
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
