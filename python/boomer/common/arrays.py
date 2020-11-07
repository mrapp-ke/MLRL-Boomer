#!/usr/bin/python

"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides utility functions for handling arrays.
"""
import numpy as np
from scipy.sparse import issparse


def enforce_dense(a, order: str):
    """
    Converts a given array into a `np.ndarray`, if necessary, and enforces a specific memory layout.

    :param a:       A `np.ndarray` or `scipy.sparse.matrix` to be converted
    :param order:   The memory layout to be used. Must be `C` or `F`
    :return:        A `np.ndarray` that uses the given memory layout
    """
    if issparse(a):
        return a.toarray(order=order)
    else:
        return np.require(a, requirements=[order])
