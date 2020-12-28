from boomer.common._types cimport uint8, uint32, float32

from libcpp.memory cimport unique_ptr, shared_ptr


cdef extern from "cpp/input/label_matrix.h" nogil:

    cdef cppclass ILabelMatrix:
        pass


    cdef cppclass IRandomAccessLabelMatrix(ILabelMatrix):
        pass


cdef extern from "cpp/input/label_matrix_dense.h" nogil:

    cdef cppclass DenseLabelMatrixImpl"DenseLabelMatrix"(IRandomAccessLabelMatrix):

        # Constructors:

        DenseLabelMatrixImpl(uint32 numExamples, uint32 numLabels, const uint8* y) except +


cdef extern from "cpp/input/label_matrix_dok.h" nogil:

    cdef cppclass DokLabelMatrixImpl"DokLabelMatrix"(IRandomAccessLabelMatrix):

        # Constructors:

        DokLabelMatrixImpl(uint32 numExamples, uint32 numLabels) except +

        # Functions:

        void setValue(uint32 exampleIndex, uint32 rowIndex)


cdef extern from "cpp/input/feature_matrix.h" nogil:

    cdef cppclass IFeatureMatrix:
        pass


cdef extern from "cpp/input/feature_matrix_c_contiguous.h" nogil:

    cdef cppclass CContiguousFeatureMatrixImpl"CContiguousFeatureMatrix":

        # Constructors:

        CContiguousFeatureMatrixImpl(uint32 numExamples, uint32 numFeatures, const float32* x) except +


cdef extern from "cpp/input/feature_matrix_csc.h" nogil:

    cdef cppclass CscFeatureMatrixImpl"CscFeatureMatrix"(IFeatureMatrix):

        # Constructors:

        CscFeatureMatrixImpl(uint32 numExamples, uint32 numFeatures, const float32* xData, const uint32* xRowIndices,
                             const uint32* xColIndices) except +


cdef extern from "cpp/input/feature_matrix_csr.h" nogil:

    cdef cppclass CsrFeatureMatrixImpl"CsrFeatureMatrix":

        # Constructors:

        CsrFeatureMatrixImpl(uint32 numExamples, uint32 numFeatures, const float32* xData, const uint32* xRowIndices,
                             const uint32 xColIndices) except +


cdef extern from "cpp/input/feature_matrix_fortran_contiguous.h" nogil:

    cdef cppclass FortranContiguousFeatureMatrixImpl"FortranContiguousFeatureMatrix"(IFeatureMatrix):

        # Constructors:

        FortranContiguousFeatureMatrixImpl(uint32 numExamples, uint32 numFeatures, const float32* x) except +


cdef extern from "cpp/input/nominal_feature_mask.h" nogil:

    cdef cppclass INominalFeatureMask:
        pass


cdef extern from "cpp/input/nominal_feature_mask_dok.h" nogil:

    cdef cppclass DokNominalFeatureMaskImpl"DokNominalFeatureMask"(INominalFeatureMask):

        # Functions:

        void setNominal(uint32 featureIndex)


cdef class LabelMatrix:

    # Attributes:

    cdef shared_ptr[ILabelMatrix] label_matrix_ptr


cdef class RandomAccessLabelMatrix(LabelMatrix):
    pass


cdef class DenseLabelMatrix(RandomAccessLabelMatrix):
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
