# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides type definitions and utility functions for creating arrays.
"""
from boomer.algorithm._model cimport intp, float64

from cython.view cimport array as cvarray
from cython.view cimport array_cwrapper as new_array


cdef inline cvarray array_float64(intp num_elements):
    """
    Creates and returns a new C-contiguous array of dtype `float64`, shape `(length)`.

    :param num_elements:    The number of elements in the array
    :return:                The array that has been created
    """
    cdef tuple shape = tuple([num_elements])
    cdef intp itemsize = sizeof(float64)
    cdef char* format = 'd'
    cdef char* mode = 'c'
    cdef char* buf = NULL
    cdef cvarray array = new_array(shape, itemsize, format, mode, buf)
    return array
