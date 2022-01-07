from mlrl.common.cython._types cimport uint8, uint32, float32, float64

from libcpp cimport bool
from libcpp.memory cimport unique_ptr


cdef extern from "common/input/label_vector.hpp" nogil:

    cdef cppclass LabelVector:

        ctypedef uint32* iterator

        ctypedef const uint32* const_iterator

        # Functions:

        iterator begin()

        iterator end()

        const_iterator cbegin()

        const_iterator cend()

        uint32 getNumElements()


cdef extern from "common/input/label_matrix.hpp" nogil:

    cdef cppclass ILabelMatrix:

        # Functions:

        uint32 getNumRows()

        uint32 getNumCols()


cdef extern from "common/input/label_matrix_row_wise.hpp" nogil:

    cdef cppclass IRowWiseLabelMatrix(ILabelMatrix):

        # Functions:

        float64 calculateLabelCardinality()

        unique_ptr[LabelVector] createLabelVector(uint32 row)


cdef extern from "common/input/label_matrix_c_contiguous.hpp" nogil:

    cdef cppclass ICContiguousLabelMatrix(IRowWiseLabelMatrix):
        pass


    unique_ptr[ICContiguousLabelMatrix] createCContiguousLabelMatrix(uint32 numRows, uint32 numCols, const uint8* array)


cdef extern from "common/input/label_matrix_csr.hpp" nogil:

    cdef cppclass ICsrLabelMatrix(IRowWiseLabelMatrix):
        pass


    unique_ptr[ICsrLabelMatrix] createCsrLabelMatrix(uint32 numRows, uint32 numCols, uint32* rowIndices,
                                                     uint32* colIndices)


cdef extern from "common/input/feature_matrix.hpp" nogil:

    cdef cppclass IFeatureMatrix:

        # Functions:

        uint32 getNumRows()

        uint32 getNumCols()


cdef extern from "common/input/feature_matrix_column_wise.hpp" nogil:

    cdef cppclass IColumnWiseFeatureMatrix(IFeatureMatrix):
        pass


cdef extern from "common/input/feature_matrix_fortran_contiguous.hpp" nogil:

    cdef cppclass IFortranContiguousFeatureMatrix(IColumnWiseFeatureMatrix):
        pass


    unique_ptr[IFortranContiguousFeatureMatrix] createFortranContiguousFeatureMatrix(uint32 numRows, uint32 numCols,
                                                                                     const float32* array)


cdef extern from "common/input/feature_matrix_csc.hpp" nogil:

    cdef cppclass ICscFeatureMatrix(IColumnWiseFeatureMatrix):
        pass


    unique_ptr[ICscFeatureMatrix] createCscFeatureMatrix(uint32 numRows, uint32 numCols, const float32* data,
                                                         uint32* rowIndices, uint32* colIndices)


cdef extern from "common/input/feature_matrix_row_wise.hpp" nogil:

    cdef cppclass IRowWiseFeatureMatrix(IFeatureMatrix):
        pass


cdef extern from "common/input/feature_matrix_c_contiguous.hpp" nogil:

    cdef cppclass CContiguousFeatureMatrixImpl"CContiguousFeatureMatrix"(IRowWiseFeatureMatrix):

        # Constructors:

        CContiguousFeatureMatrixImpl(uint32 numRows, uint32 numCols, const float32* array)


cdef extern from "common/input/feature_matrix_csr.hpp" nogil:

    cdef cppclass CsrFeatureMatrixImpl"CsrFeatureMatrix"(IRowWiseFeatureMatrix):

        # Constructors:

        CsrFeatureMatrixImpl(uint32 numRows, uint32 numCols, const float32* data, uint32* rowIndices, uint32 colIndices)


cdef extern from "common/input/nominal_feature_mask.hpp" nogil:

    cdef cppclass INominalFeatureMask:
        pass


cdef extern from "common/input/nominal_feature_mask_equal.hpp" nogil:

    cdef cppclass IEqualNominalFeatureMask(INominalFeatureMask):
        pass


    unique_ptr[IEqualNominalFeatureMask] createEqualNominalFeatureMask(bool nominal)


cdef extern from "common/input/nominal_feature_mask_mixed.hpp" nogil:

    cdef cppclass IMixedNominalFeatureMask(INominalFeatureMask):

        # Functions:

        void setNominal(uint32 featureIndex, bool nominal)


    unique_ptr[IMixedNominalFeatureMask] createMixedNominalFeatureMask(uint32 numFeatures)


cdef class LabelMatrix:

    # Functions:

    cdef ILabelMatrix* get_label_matrix_ptr(self)


cdef class RowWiseLabelMatrix(LabelMatrix):

    # Functions:

    cdef IRowWiseLabelMatrix* get_row_wise_label_matrix_ptr(self)


cdef class CContiguousLabelMatrix(RowWiseLabelMatrix):

    # Attributes:

    cdef unique_ptr[ICContiguousLabelMatrix] label_matrix_ptr


cdef class CsrLabelMatrix(RowWiseLabelMatrix):

    # Attributes:

    cdef unique_ptr[ICsrLabelMatrix] label_matrix_ptr


cdef class FeatureMatrix:

    # Functions:

    cdef IFeatureMatrix* get_feature_matrix_ptr(self)


cdef class ColumnWiseFeatureMatrix(FeatureMatrix):

    # Functions:

    cdef IColumnWiseFeatureMatrix* get_column_wise_feature_matrix_ptr(self)


cdef class FortranContiguousFeatureMatrix(ColumnWiseFeatureMatrix):

    # Attributes:

    cdef unique_ptr[IFortranContiguousFeatureMatrix] feature_matrix_ptr


cdef class CscFeatureMatrix(ColumnWiseFeatureMatrix):

    # Attributes:

    cdef unique_ptr[ICscFeatureMatrix] feature_matrix_ptr


cdef class RowWiseFeatureMatrix(FeatureMatrix):

    # Functions:

    cdef IRowWiseFeatureMatrix* get_row_wise_feature_matrix_ptr(self)


cdef class CContiguousFeatureMatrix(RowWiseFeatureMatrix):

    # Attributes:

    cdef unique_ptr[CContiguousFeatureMatrixImpl] feature_matrix_ptr


cdef class CsrFeatureMatrix(RowWiseFeatureMatrix):

    # Attributes:

    cdef unique_ptr[CsrFeatureMatrixImpl] feature_matrix_ptr


cdef class NominalFeatureMask:

    # Functions:

    cdef INominalFeatureMask* get_nominal_feature_mask_ptr(self)


cdef class EqualNominalFeatureMask(NominalFeatureMask):

    # Attributes:

    cdef unique_ptr[IEqualNominalFeatureMask] nominal_feature_mask_ptr


cdef class MixedNominalFeatureMask(NominalFeatureMask):

    # Attributes:

    cdef unique_ptr[IMixedNominalFeatureMask] nominal_feature_mask_ptr
