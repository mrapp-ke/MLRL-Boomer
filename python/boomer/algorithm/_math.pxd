# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides utility functions for common mathematical operations.
"""
from boomer.algorithm._arrays cimport float32, float64


cdef inline divide_or_zero_float64(float64 a, float64 b):
    """
    Divides a scalar of dtype `float64` by another one. The division by zero evaluates to 0 by definition.

    :param a:   The scalar to be divided
    :param b:   The divisor
    :return:    A scalar of dtype `float64`, representing the result of a / b or 0, if b = 0
    """
    if b == 0:
        return 0
    else:
        return a / b
