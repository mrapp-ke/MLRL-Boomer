"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for representing datasets.
"""
from abc import ABC
from dataclasses import dataclass, replace
from enum import Enum
from functools import reduce
from typing import List

from scipy.sparse import lil_array

from mlrl.testbed.experiments.dataset_tabular import Attribute, AttributeType

from mlrl.util.arrays import is_sparse


class DatasetType(Enum):
    """
    Characterizes a dataset as either training or test data.
    """
    TRAINING = 'training'
    TEST = 'test'


class Dataset(ABC):
    """
    An abstract base class for all datasets.
    """


@dataclass
class TabularDataset(Dataset):
    """
    A tabular dataset consisting of two matrices `x` and `y`, storing the features of examples and their respective
    ground truth, respectively.

    Attributes:
        x:          A `lil_array`, shape `(num_examples, num_features)`, that stores the features of examples
        y:          A `lil_array`, shape `(num_examples, num_features)`, that stores the ground truth of examples
        features:   A list that contains all features in the data set
        outputs:    A list that contains all outputs in the data set
    """
    x: lil_array
    y: lil_array
    features: List[Attribute]
    outputs: List[Attribute]

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

    @property
    def has_sparse_features(self) -> bool:
        """
        True, if feature values in the dataset are sparse, False otherwise.
        """
        return is_sparse(self.x)

    @property
    def has_sparse_outputs(self) -> bool:
        """
        True, if the ground truth in the dataset is sparse, False otherwise.
        """
        return is_sparse(self.y)

    def enforce_dense_features(self) -> 'TabularDataset':
        """
        Creates and returns a copy of this dataset, where the feature values have been converted into a dense format.

        :return: The dataset that has been created
        """
        if self.has_sparse_features:
            return replace(self, x=self.x.toarray())
        return self

    def enforce_dense_outputs(self) -> 'TabularDataset':
        """
        Creates and returns a copy of this dataset, where the ground truth has been converted into a dense format.

        :return: The dataset that has been created
        """
        if self.has_sparse_outputs:
            return replace(self, y=self.y.toarray())
        return self

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
