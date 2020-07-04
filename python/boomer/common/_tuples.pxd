"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides type definitions and utility functions for tuples.
"""
from boomer.common._arrays cimport intp, float32, float64


"""
A struct that stores a value of type float32 and a corresponding index that refers to the (original) position of the
value in an array.
"""
cdef struct IndexedFloat32:
    intp index
    float32 value


"""
A struct that stores a value of type float64 and a corresponding index that refers to the (original) position of the
value in an array.
"""
cdef struct IndexedFloat64:
    intp index
    float64 value


cdef inline int compare_indexed_float32(const void* a, const void* b) nogil:
    """
    Compares the values of two structs of type `IndexedFloat32`.

    :param a:   A pointer to the first struct
    :param b:   A pointer to the second struct
    :return:    -1 if the value of the first struct is smaller than the value of the second struct, 0 if both values are
                equal, or 1 if the value of the first struct is greater than the value of the second struct
    """
    cdef float32 v1 = (<IndexedFloat32*>a).value
    cdef float32 v2 = (<IndexedFloat32*>b).value
    return -1 if v1 < v2 else (0 if v1 == v2 else 1)


cdef inline int compare_indexed_float64(const void* a, const void* b) nogil:
    """
    Compares the values of two structs of type `IndexedFloat64`.

    :param a:   A pointer to the first struct
    :param b:   A pointer to the second struct
    :return:    -1 if the value of the first struct is smaller than the value of the second struct, 0 if both values are
                equal, or 1 if the value of the first struct is greater than the value of the second struct
    """
    cdef float64 v1 = (<IndexedFloat64*>a).value
    cdef float64 v2 = (<IndexedFloat64*>b).value
    return -1 if v1 < v2 else (0 if v1 == v2 else 1)
