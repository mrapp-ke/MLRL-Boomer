"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides implementations of sparse matrices.
"""
from boomer.common._arrays cimport uint32


cdef extern from "cpp/data.h" nogil:

    cdef cppclass BinaryDokMatrix:

        # Constructors:

        BinaryDokMatrix(uint32 numRows, uint32 numCols) except +

        # Functions:

        void set(uint32 row, uint32 column)


    cdef cppclass BinaryDokVector:

        # Constructors:

        BinaryDokVector(uint32 numElements) except +

        # Functions:

        void set(uint32 pos)
