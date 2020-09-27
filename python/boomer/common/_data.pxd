"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides Cython wrappers for classes that provide access to data that is stored in matrices or vectors.
"""
from boomer.common._arrays cimport uint8, uint32

from libcpp cimport bool


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

        void setValue(uint32 row, uint32 column)


    cdef cppclass IVector:

        # Functions:

        uint32 getNumElements()


    cdef cppclass IRandomAccessVector[T](IVector):

        # Functions:

        T getValue(uint32 pos)


    cdef cppclass ISparseVector(IVector):

        # Functions:

        bool hasZeroElements()


    cdef cppclass IIndexVector(ISparseVector):

        # Functions:

        uint32 getIndex(uint32 pos)


    cdef cppclass ISparseRandomAccessVector[T](ISparseVector, IRandomAccessVector[T]):
        pass


    cdef cppclass BinaryDokVector(ISparseRandomAccessVector[uint8]):

        # Constructors:

        BinaryDokVector(uint32 numElements) except +

        # Functions:

        void setValue(uint32 pos)
