from boomer.common._arrays cimport uint8, uint32, intp
from boomer.common.statistics cimport LabelMatrix
from boomer.common.head_refinement cimport HeadCandidate
from boomer.common.rule_evaluation cimport DefaultPrediction, Prediction, LabelWisePrediction


cdef class RefinementSearch:

    cdef void update_search(self, intp example_index, uint32 weight)

    cdef void reset_search(self)

    cdef LabelWisePrediction* calculate_label_wise_prediction(self, bint uncovered, bint accumulated)

    cdef Prediction* calculate_example_wise_prediction(self, bint uncovered, bint accumulated)


cdef class DecomposableRefinementSearch(RefinementSearch):

    cdef void update_search(self, intp example_index, uint32 weight)

    cdef void reset_search(self)

    cdef LabelWisePrediction* calculate_label_wise_prediction(self, bint uncovered, bint accumulated)

    cdef Prediction* calculate_example_wise_prediction(self, bint uncovered, bint accumulated)


cdef class NonDecomposableRefinementSearch(RefinementSearch):

    cdef void update_search(self, intp example_index, uint32 weight)

    cdef void reset_search(self)

    cdef LabelWisePrediction* calculate_label_wise_prediction(self, bint uncovered, bint accumulated)

    cdef Prediction* calculate_example_wise_prediction(self, bint uncovered, bint accumulated)


cdef class Loss:

    # Functions:

    cdef DefaultPrediction* calculate_default_prediction(self, LabelMatrix label_matrix)

    cdef void reset_examples(self)

    cdef void add_sampled_example(self, intp example_index, uint32 weight)

    cdef void update_covered_example(self, intp example_index, uint32 weight, bint remove)

    cdef RefinementSearch begin_search(self, intp[::1] label_indices)

    cdef void apply_prediction(self, intp example_index, intp[::1] label_indices, HeadCandidate* head)
