"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that provide access to characteristics of binary values, associated with one or several labels.
"""
from functools import cached_property

import numpy as np

from mlrl.common.data.arrays import is_sparse

from mlrl.testbed.experiments.output.characteristics.matrix_output import OutputMatrix


class LabelMatrix(OutputMatrix):
    """
    Provides access to characteristics of binary values, associated with one or several labels, that are stored in a
    `numpy.ndarray`, `scipy.sparse.spmatrix` or `scipy.sparse.sparray`.
    """

    @cached_property
    def num_distinct_label_vectors(self) -> int:
        """
        The number of distinct label vectors in the label matrix.
        """
        values = self.values

        if is_sparse(values):
            values = values.tolil()
            return np.unique(values.rows).shape[0]

        return np.unique(values, axis=0).shape[0]

    @cached_property
    def avg_label_cardinality(self) -> float:
        """
        The average label cardinality of the label matrix.
        """
        values = self.values

        if is_sparse(values):
            values = values.tocsr()
            num_relevant_per_example = values.indptr[1:] - values.indptr[:-1]
        else:
            num_relevant_per_example = np.count_nonzero(values, axis=1)

        return np.average(num_relevant_per_example)

    @cached_property
    def avg_label_imbalance_ratio(self) -> float:
        """
        The average label imbalance ratio of the label matrix.
        """
        values = self.values

        if is_sparse(values):
            values = values.tocsc()
            num_relevant_per_label = values.indptr[1:] - values.indptr[:-1]
        else:
            num_relevant_per_label = np.count_nonzero(values, axis=0)

        num_relevant_per_label = num_relevant_per_label[num_relevant_per_label != 0]

        if num_relevant_per_label.shape[0] > 0:
            return np.average(np.max(num_relevant_per_label) / num_relevant_per_label)

        return 0.0
