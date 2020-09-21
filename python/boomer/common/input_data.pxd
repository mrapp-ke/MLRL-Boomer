from boomer.common._arrays cimport uint8, uint32, float32
from boomer.common._data cimport IVector, BinaryDokVector, IMatrix, BinaryDokMatrix
from boomer.common._tuples cimport IndexedFloat32Array

from libcpp.memory cimport shared_ptr


cdef extern from "cpp/input_data.h" nogil:

    cdef cppclass AbstractLabelMatrix(IMatrix):
        pass


    cdef cppclass AbstractRandomAccessLabelMatrix(AbstractLabelMatrix):
        pass


    cdef cppclass DenseLabelMatrixImpl(AbstractRandomAccessLabelMatrix):

        # Constructors:

        DenseLabelMatrixImpl(uint32 numExamples, uint32 numLabels, const uint8* y) except +


    cdef cppclass DokLabelMatrixImpl(AbstractRandomAccessLabelMatrix):

        # Constructors:

        DokLabelMatrixImpl(shared_ptr[BinaryDokMatrix] dokMatrixPtr) except +


    cdef cppclass AbstractFeatureMatrix(IMatrix):

        # Functions:

        void fetchSortedFeatureValues(uint32 featureIndex, IndexedFloat32Array* indexedArray)


    cdef cppclass DenseFeatureMatrixImpl(AbstractFeatureMatrix):

        # Constructors:

        DenseFeatureMatrixImpl(uint32 numExamples, uint32 numFeatures, const float32* x) except +


    cdef cppclass CscFeatureMatrixImpl(AbstractFeatureMatrix):

        # Constructors:

        CscFeatureMatrixImpl(uint32 numExamples, uint32 numFeatures, const float32* xData, const uint32* xRowIndices,
                             const uint32* xColIndices) except +


    cdef cppclass AbstractNominalFeatureSet(IVector):

        # Functions:

        uint8 get(uint32 pos)


    cdef cppclass DokNominalFeatureSetImpl(AbstractNominalFeatureSet):

        # Constructors:

        DokNominalFeatureSetImpl(shared_ptr[BinaryDokVector] dokVectorPtr) except +


cdef class LabelMatrix:

    # Attributes:

    cdef shared_ptr[AbstractLabelMatrix] label_matrix_ptr


cdef class RandomAccessLabelMatrix(LabelMatrix):
    pass


cdef class DenseLabelMatrix(RandomAccessLabelMatrix):
    pass


cdef class DokLabelMatrix(RandomAccessLabelMatrix):
    pass


cdef class FeatureMatrix:

    # Attributes:

    cdef shared_ptr[AbstractFeatureMatrix] feature_matrix_ptr


cdef class DenseFeatureMatrix(FeatureMatrix):
    pass


cdef class CscFeatureMatrix(FeatureMatrix):
    pass


cdef class NominalFeatureSet:

    # Attributes:

    cdef shared_ptr[AbstractNominalFeatureSet] nominal_feature_set_ptr


cdef class DokNominalFeatureSet(NominalFeatureSet):
    pass
