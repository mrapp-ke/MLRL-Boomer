"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides Cython wrappers for classes that provide access to data that is stored in matrices or vectors.
"""
from boomer.common._arrays cimport uint32


cdef extern from "cpp/data.h" nogil:

    cdef cppclass AbstractMatrix:

        # Functions:

        uint32 getNumRows()

        uint32 getNumCols()
