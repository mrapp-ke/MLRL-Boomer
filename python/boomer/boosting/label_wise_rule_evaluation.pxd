from boomer.common._arrays cimport intp, float64
from boomer.common.rule_evaluation cimport LabelWisePrediction


cdef class LabelWiseRuleEvaluation:

    # Attributes:

    cdef float64 l2_regularization_weight

    # Functions:

    cdef void calculate_label_wise_prediction(self, const intp[::1] label_indices,
                                              const float64[::1] total_sums_of_gradients,
                                              const float64[::1] sums_of_gradients,
                                              const float64[::1] total_sums_of_hessians,
                                              const float64[::1] sums_of_hessians, bint uncovered,
                                              LabelWisePrediction* prediction)
