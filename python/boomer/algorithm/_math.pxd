# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides utility functions for common mathematical operations.
"""
from boomer.algorithm._arrays cimport float32


cdef inline float32 average_float32(float32 a, float32 b):
    """
    Calculates and returns the arithmetic mean of two scalars of dtype `float32`.

    :param a:   The first scalar
    :param b:   The second scalar
    :return:    A scalar of dtype `float32`, representing the arithmetic mean of a and b
    """
    return (a + b) / 2
