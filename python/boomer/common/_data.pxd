"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides Cython wrappers for classes that provide access to data that is stored in matrices or vectors.
"""
from boomer.common._arrays cimport uint32


cdef extern from "cpp/data.h" nogil:

    cdef cppclass IMatrix:

        # Functions:

        uint32 getNumRows()

        uint32 getNumCols()


    cdef cppclass IRandomAccessMatrix(IMatrix):
        pass


    cdef cppclass BinaryDokMatrix(IRandomAccessMatrix):

        # Constructors:

        BinaryDokMatrix(uint32 numRows, uint32 numCols) except +

        # Functions:

        void set(uint32 row, uint32 column)


    cdef cppclass IVector:

        # Functions:

        uint32 getNumElements()


    cdef cppclass IRandomAccessVector(IVector):
        pass

    cdef cppclass BinaryDokVector(IRandomAccessVector):

        # Constructors:

        BinaryDokVector(uint32 numElements) except +

        # Functions:

        void set(uint32 pos)
