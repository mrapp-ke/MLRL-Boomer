"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides Cython wrappers for classes that provide access to data that is stored in matrices or vectors.
"""
from boomer.common._arrays cimport uint32
from boomer.common._data cimport IRandomAccessVector


cdef extern from "cpp/indices.h" nogil:

    cdef cppclass IIndexVector(IRandomAccessVector[uint32]):
        pass
