"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)
"""
from boomer.common._types cimport uint32


cdef extern from "cpp/data/view_c_contiguous.h" nogil:

    cdef cppclass CContiguousView[T]:

        # Constructors

        CContiguousView(uint32 numRows, uint32 numCols, T* array) except +
