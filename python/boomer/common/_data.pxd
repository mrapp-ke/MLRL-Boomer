"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides Cython wrappers for classes that provide access to data that is stored in matrices or vectors.
"""
from boomer.common._arrays cimport uint8, uint32


cdef extern from "cpp/data.h" nogil:

    cdef cppclass IVector:

        # Functions:

        uint32 getNumElements()


    cdef cppclass BinaryDokMatrix:

        # Constructors:

        BinaryDokMatrix(uint32 numRows, uint32 numCols) except +

        # Functions:

        void setValue(uint32 row, uint32 column)
