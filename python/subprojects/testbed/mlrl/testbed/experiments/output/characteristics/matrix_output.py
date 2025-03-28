"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that provide access to characteristics of values, associated with one or several outputs.
"""
from functools import cached_property

import numpy as np

from mlrl.common.data.arrays import is_sparse


def density(array) -> float:
    """
    Calculates and returns the density of a two-dimensional array.

    :param array:   A `numpy.ndarray` or `scipy.sparse.spmatrix` or `scipy.sparse.sparray`, shape
                    `(num_rows, num_cols)`
    :return:        The fraction of dense elements explicitly stored in the given array
    """
    num_elements = array.shape[0] * array.shape[1]
    num_dense_elements = array.nnz if is_sparse(array) else np.count_nonzero(array)
    return num_dense_elements / num_elements if num_elements > 0 else 0


class OutputMatrix:
    """
    Provides access to characteristics of values, associated with one or several outputs, that are stored in a
    `numpy.ndarray`, `scipy.sparse.spmatrix` or `scipy.sparse.sparray`.
    """

    def __init__(self, values):
        """
        :param values: A `numpy.ndarray`, `scipy.sparse.spmatrix`, `scipy.sparse.sparray`, shape
                      `(num_examples, num_outputs)`, that stores the values
        """
        self.values = values

    @property
    def num_outputs(self) -> int:
        """
        The total number of outputs.
        """
        return self.values.shape[1]

    @cached_property
    def output_density(self) -> float:
        """
        The density among the values associated with the available outputs.
        """
        return density(self.values)

    @property
    def output_sparsity(self) -> float:
        """
        The sparsity among the values associated with the available outputs.
        """
        return 1 - self.output_density
