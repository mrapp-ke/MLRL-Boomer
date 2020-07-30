from boomer.common._arrays cimport uint8, uint32, intp
from boomer.common._sparse cimport BinaryDokMatrix
from boomer.common.head_refinement cimport HeadCandidate
from boomer.common.rule_evaluation cimport DefaultPrediction, Prediction, LabelWisePrediction

from libcpp cimport bool


cdef extern from "cpp/statistics.h" namespace "statistics":

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


    cdef cppclass AbstractRefinementSearch:

        # Functions:

        void updateSearch(intp statisticIndex, uint32 weight) nogil

        void resetSearch() nogil

        LabelWisePrediction* calculateLabelWisePrediction(bool uncovered, bool accumulated) nogil

        Prediction* calculateExampleWisePrediction(bool uncovered, bool accumulated) nogil


    cdef cppclass AbstractDecomposableRefinementSearch(AbstractRefinementSearch):

        # Functions:

        void updateSearch(intp statisticIndex, uint32 weight) nogil

        void resetSearch() nogil

        LabelWisePrediction* calculateLabelWisePrediction(bool uncovered, bool accumulated) nogil

        Prediction* calculateExampleWisePrediction(bool uncovered, bool accumulated) nogil


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


cdef class RefinementSearch:

    # Functions:

    cdef void update_search(self, intp statistic_index, uint32 weight)

    cdef void reset_search(self)

    cdef LabelWisePrediction* calculate_label_wise_prediction(self, bint uncovered, bint accumulated)

    cdef Prediction* calculate_example_wise_prediction(self, bint uncovered, bint accumulated)


cdef class DecomposableRefinementSearch(RefinementSearch):

    # Functions:

    cdef void update_search(self, intp statistic_index, uint32 weight)

    cdef void reset_search(self)

    cdef LabelWisePrediction* calculate_label_wise_prediction(self, bint uncovered, bint accumulated)

    cdef Prediction* calculate_example_wise_prediction(self, bint uncovered, bint accumulated)


cdef class NonDecomposableRefinementSearch(RefinementSearch):

    cdef void update_search(self, intp statistic_index, uint32 weight)

    cdef void reset_search(self)

    cdef LabelWisePrediction* calculate_label_wise_prediction(self, bint uncovered, bint accumulated)

    cdef Prediction* calculate_example_wise_prediction(self, bint uncovered, bint accumulated)


cdef class Statistics:

    # Functions:

    cdef void apply_default_prediction(self, LabelMatrix label_matrix, DefaultPrediction* default_prediction)

    cdef void reset_sampled_statistics(self)

    cdef void add_sampled_statistic(self, intp statistic_index, uint32 weight)

    cdef void reset_covered_statistics(self)

    cdef void update_covered_statistic(self, intp statistic_index, uint32 weight, bint remove)

    cdef RefinementSearch begin_search(self, intp[::1] label_indices)

    cdef void apply_prediction(self, intp statistic_index, intp[::1] label_indices, HeadCandidate* head)
