from libcpp cimport bool
from libcpp.memory cimport unique_ptr

from mlrl.common.cython._types cimport float32, uint32


cdef extern from "mlrl/common/input/feature_matrix.hpp" nogil:

    cdef cppclass IFeatureMatrix:

        # Functions:

        uint32 getNumExamples() const

        uint32 getNumFeatures() const

        bool isSparse() const


cdef extern from "mlrl/common/input/feature_matrix_column_wise.hpp" nogil:

    cdef cppclass IColumnWiseFeatureMatrix(IFeatureMatrix):
        pass


cdef extern from "mlrl/common/input/feature_matrix_fortran_contiguous.hpp" nogil:

    cdef cppclass IFortranContiguousFeatureMatrix(IColumnWiseFeatureMatrix):
        pass

    unique_ptr[IFortranContiguousFeatureMatrix] createFortranContiguousFeatureMatrix(const float32* array,
                                                                                     uint32 numRows, uint32 numCols,)


cdef extern from "mlrl/common/input/feature_matrix_csc.hpp" nogil:

    cdef cppclass ICscFeatureMatrix(IColumnWiseFeatureMatrix):
        pass

    unique_ptr[ICscFeatureMatrix] createCscFeatureMatrix(const float32* values, uint32* indices, uint32* indptr,
                                                         uint32 numRows, uint32 numCols, float32 sparseValue)


cdef extern from "mlrl/common/input/feature_matrix_row_wise.hpp" nogil:

    cdef cppclass IRowWiseFeatureMatrix(IFeatureMatrix):
        pass


cdef extern from "mlrl/common/input/feature_matrix_c_contiguous.hpp" nogil:

    cdef cppclass ICContiguousFeatureMatrix(IRowWiseFeatureMatrix):
        pass

    unique_ptr[ICContiguousFeatureMatrix] createCContiguousFeatureMatrix(const float32* array, uint32 numRows,
                                                                         uint32 numCols)


cdef extern from "mlrl/common/input/feature_matrix_csr.hpp" nogil:

    cdef cppclass ICsrFeatureMatrix(IRowWiseFeatureMatrix):
        pass

    unique_ptr[ICsrFeatureMatrix] createCsrFeatureMatrix(const float32* values, uint32* indices, uint32* indptr,
                                                         uint32 numRows, uint32 numCols, float32 sparseValue)


cdef class FeatureMatrix:

    # Functions:

    cdef IFeatureMatrix* get_feature_matrix_ptr(self)


cdef class ColumnWiseFeatureMatrix(FeatureMatrix):

    # Functions:

    cdef IColumnWiseFeatureMatrix* get_column_wise_feature_matrix_ptr(self)


cdef class FortranContiguousFeatureMatrix(ColumnWiseFeatureMatrix):

    # Attributes:

    cdef const float32[::1, :] array

    cdef unique_ptr[IFortranContiguousFeatureMatrix] feature_matrix_ptr


cdef class CscFeatureMatrix(ColumnWiseFeatureMatrix):

    # Attributes:

    cdef const float32[::1] values

    cdef uint32[::1] indices

    cdef uint32[::1] indptr

    cdef unique_ptr[ICscFeatureMatrix] feature_matrix_ptr


cdef class RowWiseFeatureMatrix(FeatureMatrix):

    # Functions:

    cdef IRowWiseFeatureMatrix* get_row_wise_feature_matrix_ptr(self)


cdef class CContiguousFeatureMatrix(RowWiseFeatureMatrix):

    # Attributes:

    cdef const float32[:, ::1] array

    cdef unique_ptr[ICContiguousFeatureMatrix] feature_matrix_ptr


cdef class CsrFeatureMatrix(RowWiseFeatureMatrix):

    # Attributes:

    cdef const float32[::1] values

    cdef const uint32[::1] indices

    cdef const uint32[::1] indptr

    cdef unique_ptr[ICsrFeatureMatrix] feature_matrix_ptr
