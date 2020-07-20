from boomer.common._arrays cimport uint32, intp, float64
from boomer.common.losses cimport LabelMatrix
from boomer.common.losses cimport Loss, RefinementSearch
from boomer.common.rule_evaluation cimport DefaultPrediction, Prediction, LabelWisePrediction
from boomer.common.head_refinement cimport HeadCandidate


cdef class CoverageLoss(Loss):

    # Attributes:

    cdef readonly float64 sum_uncovered_labels

    # Functions:

    cdef DefaultPrediction* calculate_default_prediction(self, LabelMatrix label_matrix)

    cdef void reset_examples(self)

    cdef void add_sampled_example(self, intp example_index, uint32 weight)

    cdef void update_covered_example(self, intp example_index, uint32 weight, bint remove)

    cdef RefinementSearch begin_search(self, intp[::1] label_indices)

    cdef void apply_prediction(self, intp example_index, intp[::1] label_indices, HeadCandidate* head)
