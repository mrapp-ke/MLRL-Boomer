"""
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides commonly used functions that implement mathematical operations.
"""
from boomer.common._arrays cimport intp


cdef inline intp triangular_number(intp n):
    """
    Computes and returns the n-th triangular number, i.e., the number of elements in a n times n triangle.

    :param n:   A scalar of dtype `intp`, representing the order of the triangular number
    :return:    A scalar of dtype `intp`, representing the n-th triangular number
    """
    return (n * (n + 1)) // 2
