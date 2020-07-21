"""
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides commonly used functions that implement mathematical operations.
"""
from boomer.common._arrays cimport intp, float64

from libc.math cimport pow


cdef inline float64 l2_norm_pow(float64* a, intp n):
    """
    Computes and returns the square of the L2 norm of a specific vector, i.e. the sum of the squares of its elements. To
    obtain the actual L2 norm, the square-root of the result provided by this function must be computed.

    :param a:   A pointer to an array of type `float64`, shape `(n)`, representing the elements in the vector
    :param n:   The number of elements in the array `a`
    :return:    A scalar of dtype `float64`, representing the square of the L2 of the given vector
    """
    cdef float64 result = 0
    cdef float64 tmp
    cdef intp i

    for i in range(n):
        tmp = a[i]
        tmp = pow(tmp, 2)
        result += tmp

    return result


cdef inline intp triangular_number(intp n):
    """
    Computes and returns the n-th triangular number, i.e., the number of elements in a n times n triangle.

    :param n:   A scalar of dtype `intp`, representing the order of the triangular number
    :return:    A scalar of dtype `intp`, representing the n-th triangular number
    """
    return (n * (n + 1)) // 2
