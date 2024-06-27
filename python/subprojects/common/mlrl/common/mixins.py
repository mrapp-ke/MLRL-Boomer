"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides mixins that additional functionality to machine learning algorithms.
"""
from abc import ABC
from typing import List, Optional


class OrdinalFeatureSupportMixin(ABC):
    """
    A mixin for all machine learning algorithms that natively support ordinal features.
    """

    ordinal_feature_indices: Optional[List[int]] = None

    def set_ordinal_feature_indices(self, indices):
        """
        Sets the indices of all ordinal features.

        :param indices: A `np.ndarray` or `Iterable` that stores the indices to be set
        """
        self.ordinal_feature_indices = None if indices is None else list(indices)
