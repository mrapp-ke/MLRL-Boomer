from boomer.common._arrays cimport uint8, intp, float32
from boomer.common._tuples cimport IndexedFloat32Array
from boomer.common._sparse cimport BinaryDokMatrix

from libcpp.memory cimport shared_ptr


cdef extern from "cpp/input_data.h" nogil:

    cdef cppclass AbstractLabelMatrix:

        # Attributes:

        intp numExamples_

        intp numLabels_


    cdef cppclass AbstractRandomAccessLabelMatrix(AbstractLabelMatrix):

        # Functions:

        uint8 getLabel(intp exampleIndex, intp labelIndex)


    cdef cppclass DenseLabelMatrixImpl(AbstractRandomAccessLabelMatrix):

        # Constructors:

        DenseLabelMatrixImpl(intp numExamples, intp numLabels, uint8* y) except +

        # Functions:

        uint8 getLabel(intp exampleIndex, intp labelIndex)


    cdef cppclass DokLabelMatrixImpl(AbstractRandomAccessLabelMatrix):

        # Constructors:

        DokLabelMatrixImpl(intp numExamples, intp numLabels, shared_ptr[BinaryDokMatrix] dokMatrix) except +

        # Functions:

        uint8 getLabel(intp exampleIndex, intp labelIndex)


cdef class LabelMatrix:

    # Attributes:

    cdef shared_ptr[AbstractLabelMatrix] label_matrix_ptr

    cdef readonly intp num_examples

    cdef readonly intp num_labels


cdef class RandomAccessLabelMatrix(LabelMatrix):
    pass


cdef class DenseLabelMatrix(RandomAccessLabelMatrix):
    pass


cdef class DokLabelMatrix(RandomAccessLabelMatrix):
    pass


cdef class FeatureMatrix:

    # Attributes:

    cdef readonly intp num_examples

    cdef readonly intp num_features

    # Functions:

    cdef void fetch_sorted_feature_values(self, intp feature_index, IndexedFloat32Array* indexed_array) nogil


cdef class DenseFeatureMatrix(FeatureMatrix):

    # Attributes:

    cdef const float32[::1, :] x

    # Functions:

    cdef void fetch_sorted_feature_values(self, intp feature_index, IndexedFloat32Array* indexed_array) nogil


cdef class CscFeatureMatrix(FeatureMatrix):

    # Attributes:

    cdef const float32[::1] x_data

    cdef const intp[::1] x_row_indices

    cdef const intp[::1] x_col_indices

    # Functions:

    cdef void fetch_sorted_feature_values(self, intp feature_index, IndexedFloat32Array* indexed_array) nogil
