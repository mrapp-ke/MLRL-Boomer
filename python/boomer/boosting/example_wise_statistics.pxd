from boomer.common._arrays cimport uint32, intp, float64
from boomer.common.statistics cimport LabelMatrix, RefinementSearch, NonDecomposableRefinementSearch
from boomer.common.head_refinement cimport HeadCandidate
from boomer.common.rule_evaluation cimport DefaultPrediction, Prediction, LabelWisePrediction
from boomer.boosting.gradient_statistics cimport GradientStatistics
from boomer.boosting.losses cimport ExampleWiseLossFunction


cdef class ExampleWiseRefinementSearch(NonDecomposableRefinementSearch):

    # Attributes:

    cdef const intp[::1] label_indices

    cdef const float64[:, ::1] gradients

    cdef const float64[::1] total_sums_of_gradients

    cdef float64[::1] sums_of_gradients

    cdef const float64[:, ::1] hessians

    cdef const float64[::1] total_sums_of_hessians

    cdef float64[::1] sums_of_hessians

    # Functions:

    cdef void update_search(self, intp statistic_index, uint32 weight)

    cdef void reset_search(self)

    cdef LabelWisePrediction* calculate_label_wise_prediction(self, bint uncovered, bint accumulated)

    cdef Prediction* calculate_example_wise_prediction(self, bint uncovered, bint accumulated)


cdef class ExampleWiseStatistics(GradientStatistics):

    # Attributes:

    cdef ExampleWiseLossFunction loss_function

    cdef LabelMatrix label_matrix

    cdef float64[:, ::1] current_scores

    cdef float64[:, ::1] gradients

    cdef float64[::1] total_sums_of_gradients

    cdef float64[:, ::1] hessians

    cdef float64[::1] total_sums_of_hessians

    # Functions:

    cdef void apply_default_prediction(self, LabelMatrix label_matrix, DefaultPrediction* default_prediction)

    cdef void reset_statistics(self)

    cdef void add_sampled_statistic(self, intp statistic_index, uint32 weight)

    cdef void update_covered_statistic(self, intp statistic_index, uint32 weight, bint remove)

    cdef RefinementSearch begin_search(self, intp[::1] label_indices)

    cdef void apply_prediction(self, intp statistic_index, intp[::1] label_indices, HeadCandidate* head)
