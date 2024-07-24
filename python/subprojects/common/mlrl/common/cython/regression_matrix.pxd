from libcpp.memory cimport unique_ptr

from mlrl.common.cython._types cimport float32, uint32
from mlrl.common.cython.output_matrix cimport IOutputMatrix, OutputMatrix


cdef extern from "mlrl/common/input/regression_matrix_row_wise.hpp" nogil:

    cdef cppclass IRowWiseRegressionMatrix(IOutputMatrix):
        pass


cdef extern from "mlrl/common/input/regression_matrix_c_contiguous.hpp" nogil:

    cdef cppclass ICContiguousRegressionMatrix(IRowWiseRegressionMatrix):
        pass


    unique_ptr[ICContiguousRegressionMatrix] createCContiguousRegressionMatrix(const float32* array, uint32 numRows,
                                                                               uint32 numCols)


cdef extern from "mlrl/common/input/regression_matrix_csr.hpp" nogil:

    cdef cppclass ICsrRegressionMatrix(IRowWiseRegressionMatrix):
        pass


    unique_ptr[ICsrRegressionMatrix] createCsrRegressionMatrix(float32* values, uint32* indices, uint32* indptr,
                                                               uint32 numRows, uint32 numCols)


cdef class RowWiseRegressionMatrix(OutputMatrix):

    # Functions:

    cdef IRowWiseRegressionMatrix* get_row_wise_regression_matrix_ptr(self)


cdef class CContiguousRegressionMatrix(RowWiseRegressionMatrix):

    # Attributes:

    cdef const float32[:, ::1] array

    cdef unique_ptr[ICContiguousRegressionMatrix] regression_matrix_ptr


cdef class CsrRegressionMatrix(RowWiseRegressionMatrix):

    # Attributes:

    cdef float32[::1] values

    cdef uint32[::1] indices

    cdef uint32[::1] indptr

    cdef unique_ptr[ICsrRegressionMatrix] regression_matrix_ptr
