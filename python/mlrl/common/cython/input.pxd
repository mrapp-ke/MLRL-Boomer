from mlrl.common.cython._types cimport uint8, uint32, float32

from libcpp cimport bool
from libcpp.memory cimport unique_ptr, shared_ptr


cdef extern from "common/input/label_matrix.hpp" nogil:

    cdef cppclass ILabelMatrix:

        # Functions:

        uint32 getNumRows()

        uint32 getNumCols()


    cdef cppclass IRandomAccessLabelMatrix(ILabelMatrix):

        # Functions:

        uint8 getValue(uint32 row, uint32 col)


cdef extern from "common/input/label_matrix_c_contiguous.hpp" nogil:

    cdef cppclass CContiguousLabelMatrixImpl"CContiguousLabelMatrix"(IRandomAccessLabelMatrix):

        # Constructors:

        CContiguousLabelMatrixImpl(uint32 numRows, uint32 numCols, uint8* array) except +


cdef extern from "common/input/label_matrix_csr.hpp" nogil:

    cdef cppclass CsrLabelMatrixImpl"CsrLabelMatrix"(ILabelMatrix):

        # Constructors:

        CsrLabelMatrixImpl(uint32 numRows, uint32 numCols, const uint32* rowIndices, const uint32* colIndices) except +


cdef extern from "common/input/label_matrix_dok.hpp" nogil:

    cdef cppclass DokLabelMatrixImpl"DokLabelMatrix"(IRandomAccessLabelMatrix):

        # Constructors:

        DokLabelMatrixImpl(uint32 numRows, uint32 numCols) except +

        # Functions:

        void setValue(uint32 exampleIndex, uint32 rowIndex)


cdef extern from "common/input/label_vector.hpp" nogil:

    cdef cppclass LabelVector:

        ctypedef uint32* index_iterator

        ctypedef const uint32* index_const_iterator

        # Constructors:

        LabelVector(uint32 numElements)

        # Functions:

        index_iterator indices_begin()

        index_iterator indices_end()

        index_const_iterator indices_cbegin()

        index_const_iterator indices_cend()

        uint32 getNumElements()

        void setNumElements(uint32 numElements, bool freeMemory)


cdef extern from "common/input/feature_matrix.hpp" nogil:

    cdef cppclass IFeatureMatrix:
        pass


cdef extern from "common/input/feature_matrix_c_contiguous.hpp" nogil:

    cdef cppclass CContiguousFeatureMatrixImpl"CContiguousFeatureMatrix":

        # Constructors:

        CContiguousFeatureMatrixImpl(uint32 numRows, uint32 numCols, float32* array) except +

        # Functions:

        uint32 getNumRows()


cdef extern from "common/input/feature_matrix_fortran_contiguous.hpp" nogil:

    cdef cppclass FortranContiguousFeatureMatrixImpl"FortranContiguousFeatureMatrix"(IFeatureMatrix):

        # Constructors:

        FortranContiguousFeatureMatrixImpl(uint32 numRows, uint32 numCols, float32* array) except +


cdef extern from "common/input/feature_matrix_csc.hpp" nogil:

    cdef cppclass CscFeatureMatrixImpl"CscFeatureMatrix"(IFeatureMatrix):

        # Constructors:

        CscFeatureMatrixImpl(uint32 numRows, uint32 numCols, const float32* data, const uint32* rowIndices,
                             const uint32* colIndices) except +


cdef extern from "common/input/feature_matrix_csr.hpp" nogil:

    cdef cppclass CsrFeatureMatrixImpl"CsrFeatureMatrix":

        # Constructors:

        CsrFeatureMatrixImpl(uint32 numRows, uint32 numCols, const float32* data, const uint32* rowIndices,
                             const uint32 colIndices) except +

        # Functions:

        uint32 getNumRows()


cdef extern from "common/input/nominal_feature_mask.hpp" nogil:

    cdef cppclass INominalFeatureMask:
        pass


cdef extern from "common/input/nominal_feature_mask_dok.hpp" nogil:

    cdef cppclass DokNominalFeatureMaskImpl"DokNominalFeatureMask"(INominalFeatureMask):

        # Functions:

        void setNominal(uint32 featureIndex)


cdef extern from "common/input/nominal_feature_mask_equal.hpp" nogil:

    cdef cppclass EqualNominalFeatureMaskImpl"EqualNominalFeatureMask"(INominalFeatureMask):

        # Constructors:

        EqualNominalFeatureMaskImpl(bool nominal) except +


cdef class LabelMatrix:

    # Attributes:

    cdef shared_ptr[ILabelMatrix] label_matrix_ptr


cdef class RandomAccessLabelMatrix(LabelMatrix):
    pass


cdef class CContiguousLabelMatrix(RandomAccessLabelMatrix):
    pass


cdef class CsrLabelMatrix(LabelMatrix):
    pass


cdef class DokLabelMatrix(RandomAccessLabelMatrix):
    pass


cdef class FeatureMatrix:

    # Attributes:

    cdef shared_ptr[IFeatureMatrix] feature_matrix_ptr


cdef class FortranContiguousFeatureMatrix(FeatureMatrix):
    pass


cdef class CscFeatureMatrix(FeatureMatrix):
    pass


cdef class CContiguousFeatureMatrix:

    # Attributes:

    cdef shared_ptr[CContiguousFeatureMatrixImpl] feature_matrix_ptr


cdef class CsrFeatureMatrix:

    # Attributes:

    cdef shared_ptr[CsrFeatureMatrixImpl] feature_matrix_ptr


cdef class NominalFeatureMask:

    # Attributes:

    cdef shared_ptr[INominalFeatureMask] nominal_feature_mask_ptr


cdef class DokNominalFeatureMask(NominalFeatureMask):
    pass


cdef class EqualNominalFeatureMask(NominalFeatureMask):
    pass
