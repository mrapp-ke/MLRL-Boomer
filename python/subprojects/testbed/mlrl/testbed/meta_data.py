"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for representing the meta-data of datasets.
"""
from dataclasses import dataclass
from functools import reduce
from typing import List

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
