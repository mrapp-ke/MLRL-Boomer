"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utility functions for handling arrays.
"""
import numpy as np
from scipy.sparse import issparse


def enforce_dense(a, order: str, dtype) -> np.ndarray:
    """
    Converts a given array into a `np.ndarray`, if necessary, and enforces a specific memory layout and data type to be
    used.

    :param a:       A `np.ndarray` or `scipy.sparse.matrix` to be converted
    :param order:   The memory layout to be used. Must be `C` or `F`
    :param dtype:   The data type to be used
    :return:        A `np.ndarray` that uses the given memory layout and data type
    """
    if issparse(a):
        return np.require(a.toarray(order=order), dtype=dtype)
    else:
        return np.require(a, dtype=dtype, requirements=[order])
