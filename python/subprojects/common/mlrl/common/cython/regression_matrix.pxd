from libcpp.memory cimport unique_ptr

from mlrl.common.cython._types cimport float32, uint32
from mlrl.common.cython.output_matrix cimport IOutputMatrix, OutputMatrix


cdef extern from "mlrl/common/input/regression_matrix_c_contiguous.hpp" nogil:

    cdef cppclass ICContiguousRegressionMatrix(IOutputMatrix):
        pass


    unique_ptr[ICContiguousRegressionMatrix] createCContiguousRegressionMatrix(const float32* array, uint32 numRows,
                                                                               uint32 numCols)


cdef class CContiguousRegressionMatrix(OutputMatrix):

    # Attributes:

    cdef const float32[:, ::1] array

    cdef unique_ptr[ICContiguousRegressionMatrix] regression_matrix_ptr
