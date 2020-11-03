from boomer.common._arrays cimport uint8, uint32, float32
from boomer.common._data cimport IRandomAccessVector, BinaryDokVector, BinaryDokMatrix

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

        DokLabelMatrixImpl(unique_ptr[BinaryDokMatrix] matrixPtr) except +


    cdef cppclass IFeatureMatrix:
        pass


    cdef cppclass DenseFeatureMatrixImpl(IFeatureMatrix):

        # Constructors:

        DenseFeatureMatrixImpl(uint32 numExamples, uint32 numFeatures, const float32* x) except +


    cdef cppclass CscFeatureMatrixImpl(IFeatureMatrix):

        # Constructors:

        CscFeatureMatrixImpl(uint32 numExamples, uint32 numFeatures, const float32* xData, const uint32* xRowIndices,
                             const uint32* xColIndices) except +


    cdef cppclass INominalFeatureVector(IRandomAccessVector[uint8]):
        pass


    cdef cppclass DokNominalFeatureVectorImpl(INominalFeatureVector):

        # Constructors:

        DokNominalFeatureVectorImpl(unique_ptr[BinaryDokVector] dokVectorPtr) except +


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


cdef class NominalFeatureVector:

    # Attributes:

    cdef shared_ptr[INominalFeatureVector] nominal_feature_vector_ptr


cdef class DokNominalFeatureVector(NominalFeatureVector):
    pass
