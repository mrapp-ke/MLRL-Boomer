"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides utility functions for common mathematical operations.
"""
from boomer.algorithm._arrays cimport intp, float64

from libc.math cimport pow


cdef inline float64 l2_norm_pow(float64[::1] a):
    """
    Computes and returns the square of the L2 norm of a specific vector, i.e. the sum of the squares of its elements. To 
    obtain the actual L2 norm, the square-root of the result provided by this function must be computed.
    
    :param a:   An array of dtype `float64`, shape (n), representing a vector
    :return:    A scalar of dtype `float64`, representing the square of the L2 of the given vector
    """
    cdef float64 result = 0
    cdef intp n = a.shape[0]
    cdef float64 tmp
    cdef intp i

    for i in range(n):
        tmp = a[i]
        tmp = pow(tmp, 2)
        result += tmp

    return result
