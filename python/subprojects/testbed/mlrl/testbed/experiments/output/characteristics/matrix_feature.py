"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that provide access to characteristics of feature values, associated with one or several features.
"""
from functools import cached_property

from mlrl.testbed.dataset import AttributeType, Dataset
from mlrl.testbed.experiments.output.characteristics.matrix_output import density


class FeatureMatrix:
    """
    Provides access to characteristics of feature values, associated with one or several features, that are stored in a
    dataset.
    """

    def __init__(self, dataset: Dataset):
        """
        :param dataset: The dataset
        """
        self.dataset = dataset

    @property
    def num_examples(self) -> int:
        """
        The total number of examples.
        """
        return self.dataset.num_examples

    @property
    def num_features(self) -> int:
        """
        The total number of features.
        """
        return self.dataset.num_features

    @cached_property
    def num_nominal_features(self) -> int:
        """
        The total number of nominal features.
        """
        return self.dataset.get_num_features(AttributeType.NOMINAL)

    @cached_property
    def num_ordinal_features(self) -> int:
        """
        The total number of ordinal features.
        """
        return self.dataset.get_num_features(AttributeType.ORDINAL)

    @cached_property
    def num_numerical_features(self) -> int:
        """
        The total number of numerical features.
        """
        return self.dataset.get_num_features(AttributeType.NUMERICAL)

    @cached_property
    def feature_density(self) -> float:
        """
        The feature density.
        """
        return density(self.dataset.x)

    @property
    def feature_sparsity(self) -> float:
        """
        The feature sparsity.
        """
        return 1 - self.feature_density
