"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides implementations of sparse matrices.
"""
from boomer.common._arrays cimport uint8, uint32


cdef extern from "cpp/sparse.h":

    cdef cppclass BinaryDokMatrix:

        # Constructors:

        BinaryDokMatrix() except +

        # Functions:

        void addValue(uint32 row, uint32 column) nogil

        uint8 getValue(uint32 row, uint32 column) nogil
