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
