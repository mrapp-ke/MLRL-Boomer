from boomer.common._arrays cimport uint8, intp, float32
from boomer.common._tuples cimport IndexedFloat32Array
from boomer.common._sparse cimport BinaryDokMatrix

from libcpp.memory cimport shared_ptr


cdef extern from "cpp/input_data.h" nogil:

    cdef cppclass AbstractLabelMatrix:

        # Attributes:

        intp numExamples_

        intp numLabels_

        # Functions:

        uint8 getLabel(intp exampleIndex, intp labelIndex)


    cdef cppclass DenseLabelMatrixImpl(AbstractLabelMatrix):

        # Constructors:

        DenseLabelMatrixImpl(intp numExamples, intp numLabels, uint8* y) except +

        # Functions:

        uint8 getLabel(intp exampleIndex, intp labelIndex)


    cdef cppclass DokLabelMatrixImpl(AbstractLabelMatrix):

        # Constructors:

        DokLabelMatrixImpl(intp numExamples, intp numLabels, BinaryDokMatrix* dokMatrix) except +

        # Functions:

        uint8 getLabel(intp exampleIndex, intp labelIndex)


cdef class LabelMatrix:

    # Attributes:

    cdef readonly intp num_examples

    cdef readonly intp num_labels

    cdef shared_ptr[AbstractLabelMatrix] label_matrix_ptr


cdef class DenseLabelMatrix(LabelMatrix):
    pass


cdef class DokLabelMatrix(LabelMatrix):
    pass


cdef class FeatureMatrix:

    # Attributes:

    cdef readonly intp num_examples

    cdef readonly intp num_features

    # Functions:

    cdef IndexedFloat32Array* get_sorted_feature_values(self, intp feature_index) nogil


cdef class DenseFeatureMatrix(FeatureMatrix):

    # Attributes:

    cdef const float32[::1, :] x

    # Functions:

    cdef IndexedFloat32Array* get_sorted_feature_values(self, intp feature_index) nogil


cdef class CscFeatureMatrix(FeatureMatrix):

    # Attributes:

    cdef const float32[::1] x_data

    cdef const intp[::1] x_row_indices

    cdef const intp[::1] x_col_indices

    # Functions:

    cdef IndexedFloat32Array* get_sorted_feature_values(self, intp feature_index) nogil
