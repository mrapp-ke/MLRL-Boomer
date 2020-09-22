from boomer.common._arrays cimport uint8, uint32, float32
from boomer.common._data cimport IVector, BinaryDokVector, IMatrix, BinaryDokMatrix
from boomer.common._tuples cimport IndexedFloat32Array

from libcpp.memory cimport shared_ptr


cdef extern from "cpp/input_data.h" nogil:

    cdef cppclass ILabelMatrix(IMatrix):
        pass


    cdef cppclass IRandomAccessLabelMatrix(ILabelMatrix):
        pass


    cdef cppclass DenseLabelMatrixImpl(IRandomAccessLabelMatrix):

        # Constructors:

        DenseLabelMatrixImpl(uint32 numExamples, uint32 numLabels, const uint8* y) except +


    cdef cppclass DokLabelMatrixImpl(IRandomAccessLabelMatrix):

        # Constructors:

        DokLabelMatrixImpl(shared_ptr[BinaryDokMatrix] dokMatrixPtr) except +


    cdef cppclass IFeatureMatrix(IMatrix):

        # Functions:

        void fetchSortedFeatureValues(uint32 featureIndex, IndexedFloat32Array* indexedArray)


    cdef cppclass DenseFeatureMatrixImpl(IFeatureMatrix):

        # Constructors:

        DenseFeatureMatrixImpl(uint32 numExamples, uint32 numFeatures, const float32* x) except +


    cdef cppclass CscFeatureMatrixImpl(IFeatureMatrix):

        # Constructors:

        CscFeatureMatrixImpl(uint32 numExamples, uint32 numFeatures, const float32* xData, const uint32* xRowIndices,
                             const uint32* xColIndices) except +


    cdef cppclass INominalFeatureSet(IVector):

        # Functions:

        uint8 get(uint32 pos)


    cdef cppclass DokNominalFeatureSetImpl(INominalFeatureSet):

        # Constructors:

        DokNominalFeatureSetImpl(shared_ptr[BinaryDokVector] dokVectorPtr) except +


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


cdef class NominalFeatureSet:

    # Attributes:

    cdef shared_ptr[INominalFeatureSet] nominal_feature_set_ptr


cdef class DokNominalFeatureSet(NominalFeatureSet):
    pass
