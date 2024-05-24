"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utility functions for handling arrays.
"""
import numpy as np

from scipy.sparse import issparse


def enforce_dense(array, order: str, dtype, sparse_value=0) -> np.ndarray:
    """
    Converts a given array into a `np.ndarray`, if necessary, and enforces a specific memory layout and data type to be
    used.

    :param array:           A `np.ndarray` or `scipy.sparse.matrix` to be converted
    :param order:           The memory layout to be used. Must be `C` or `F`
    :param dtype:           The data type to be used
    :param sparse_value:    The value that should be used for sparse elements in the given array
    :return:                A `np.ndarray` that uses the given memory layout and data type
    """
    if issparse(array):
        if sparse_value != 0:
            dense_array = np.full(shape=array.shape, fill_value=sparse_value, dtype=dtype, order=order)
            dense_array[array.nonzero()] = 0
            dense_array += array
            return np.asarray(dense_array, dtype=dtype, order=order)
        return np.require(array.toarray(order=order), dtype=dtype)
    return np.require(array, dtype=dtype, requirements=[order])


def enforce_2d(array: np.ndarray) -> np.ndarray:
    """
    Converts a given `np.ndarray` into a two-dimensional array if it is one-dimensional.

    :param array:   A `np.ndarray` to be converted
    :return:        A `np.ndarray` with at least two dimensions
    """
    if array.ndim == 1:
        return np.expand_dims(array, axis=1)
    return array
