from boomer.common._arrays cimport uint8, intp
from boomer.common._sparse cimport BinaryDokMatrix


cdef extern from "cpp/input_data.h" namespace "input":

    cdef cppclass AbstractLabelMatrix:

        # Attributes:

        intp numExamples_

        intp numLabels_

        # Functions:

        uint8 getLabel(intp exampleIndex, intp labelIndex) nogil


    cdef cppclass DenseLabelMatrixImpl(AbstractLabelMatrix):

        # Constructors:

        DenseLabelMatrixImpl(intp numExamples, intp numLabels, uint8* y) except +

        # Functions:

        uint8 getLabel(intp exampleIndex, intp labelIndex) nogil


    cdef cppclass DokLabelMatrixImpl(AbstractLabelMatrix):

        # Constructors:

        DokLabelMatrixImpl(intp numExamples, intp numLabels, BinaryDokMatrix* dokMatrix) except +

        # Functions:

        uint8 getLabel(intp exampleIndex, intp labelIndex) nogil


cdef class LabelMatrix:

    # Attributes:

    cdef AbstractLabelMatrix* label_matrix

    cdef readonly intp num_examples

    cdef readonly intp num_labels

    # Functions:

    cdef uint8 get_label(self, intp example_index, intp label_index) nogil


cdef class DenseLabelMatrix(LabelMatrix):

    # Functions:

    cdef uint8 get_label(self, intp example_index, intp label_index) nogil


cdef class DokLabelMatrix(LabelMatrix):

    # Functions:

    cdef uint8 get_label(self, intp example_index, intp label_index) nogil
