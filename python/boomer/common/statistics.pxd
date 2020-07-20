from boomer.common._arrays cimport uint8, uint32, intp
from boomer.common._sparse cimport BinaryDokMatrix
from boomer.common.head_refinement cimport HeadCandidate
from boomer.common.rule_evaluation cimport Prediction, LabelWisePrediction


cdef class LabelMatrix:

    # Attributes:

    cdef readonly intp num_examples

    cdef readonly intp num_labels

    # Functions:

    cdef uint8 get_label(self, intp example_index, intp label_index)


cdef class DenseLabelMatrix(LabelMatrix):

    # Attributes:

    cdef const uint8[:, ::1] y

    # Functions:

    cdef uint8 get_label(self, intp example_index, intp label_index)


cdef class SparseLabelMatrix(LabelMatrix):

    # Attributes:

    cdef BinaryDokMatrix* dok_matrix

    # Functions:

    cdef uint8 get_label(self, intp example_index, intp label_index)


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

    cdef void reset_examples(self)

    cdef void add_sampled_example(self, intp statistic_index, uint32 weight)

    cdef void update_covered_statistic(self, intp statistic_index, uint32 weight, bint remove)

    cdef RefinementSearch begin_search(self, intp[::1] label_indices)

    cdef void apply_prediction(self, intp statistic_index, intp[::1] label_indices, HeadCandidate* head)
