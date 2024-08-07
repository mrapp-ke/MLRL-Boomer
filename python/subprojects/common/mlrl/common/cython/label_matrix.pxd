from libcpp.memory cimport unique_ptr

from mlrl.common.cython._types cimport float32, uint8, uint32
from mlrl.common.cython.output_matrix cimport IOutputMatrix, OutputMatrix


cdef extern from "mlrl/common/input/label_matrix_row_wise.hpp" nogil:

    cdef cppclass IRowWiseLabelMatrix(IOutputMatrix):

        # Functions:

        float32 calculateLabelCardinality() const


cdef extern from "mlrl/common/input/label_matrix_c_contiguous.hpp" nogil:

    cdef cppclass ICContiguousLabelMatrix(IRowWiseLabelMatrix):
        pass


    unique_ptr[ICContiguousLabelMatrix] createCContiguousLabelMatrix(const uint8* array, uint32 numRows, uint32 numCols)


cdef extern from "mlrl/common/input/label_matrix_csr.hpp" nogil:

    cdef cppclass ICsrLabelMatrix(IRowWiseLabelMatrix):
        pass


    unique_ptr[ICsrLabelMatrix] createCsrLabelMatrix(uint32* indices, uint32* indptr, uint32 numRows, uint32 numCols)


cdef class RowWiseLabelMatrix(OutputMatrix):

    # Functions:

    cdef IRowWiseLabelMatrix* get_row_wise_label_matrix_ptr(self)


cdef class CContiguousLabelMatrix(RowWiseLabelMatrix):

    # Attributes:

    cdef const uint8[:, ::1] array

    cdef unique_ptr[ICContiguousLabelMatrix] label_matrix_ptr


cdef class CsrLabelMatrix(RowWiseLabelMatrix):

    # Attributes:

    cdef uint32[::1] indices

    cdef uint32[::1] indptr

    cdef unique_ptr[ICsrLabelMatrix] label_matrix_ptr
