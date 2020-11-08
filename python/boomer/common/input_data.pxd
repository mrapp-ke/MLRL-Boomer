from boomer.common._types cimport uint8, uint32, float32

from libcpp.memory cimport unique_ptr, shared_ptr


cdef extern from "cpp/input_data.h" nogil:

    cdef cppclass ILabelMatrix:
        pass


    cdef cppclass IRandomAccessLabelMatrix(ILabelMatrix):
        pass


    cdef cppclass DenseLabelMatrixImpl(IRandomAccessLabelMatrix):

        # Constructors:

        DenseLabelMatrixImpl(uint32 numExamples, uint32 numLabels, const uint8* y) except +


    cdef cppclass DokLabelMatrixImpl(IRandomAccessLabelMatrix):

        # Constructors:

        DokLabelMatrixImpl(uint32 numExamples, uint32 numLabels) except +

        # Functions:

        void setValue(uint32 exampleIndex, uint32 rowIndex)


    cdef cppclass IFeatureMatrix:
        pass


    cdef cppclass DenseFeatureMatrixImpl(IFeatureMatrix):

        # Constructors:

        DenseFeatureMatrixImpl(uint32 numExamples, uint32 numFeatures, const float32* x) except +


    cdef cppclass CscFeatureMatrixImpl(IFeatureMatrix):

        # Constructors:

        CscFeatureMatrixImpl(uint32 numExamples, uint32 numFeatures, const float32* xData, const uint32* xRowIndices,
                             const uint32* xColIndices) except +


    cdef cppclass INominalFeatureMask:
        pass


    cdef cppclass DokNominalFeatureMaskImpl(INominalFeatureMask):

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


cdef class DenseFeatureMatrix(FeatureMatrix):
    pass


cdef class CscFeatureMatrix(FeatureMatrix):
    pass


cdef class NominalFeatureMask:

    # Attributes:

    cdef shared_ptr[INominalFeatureMask] nominal_feature_mask_ptr


cdef class DokNominalFeatureMask(NominalFeatureMask):
    pass
