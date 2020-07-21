from boomer.common._arrays cimport intp, float64
from boomer.common.statistics cimport LabelMatrix
from boomer.common.rule_evaluation cimport DefaultPrediction, LabelWisePrediction, DefaultRuleEvaluation
from boomer.boosting.losses cimport LabelWiseLossFunction


cdef class LabelWiseDefaultRuleEvaluation(DefaultRuleEvaluation):

    # Attributes:

    cdef LabelWiseLossFunction loss_function

    cdef float64 l2_regularization_weight

    # Functions:

    cdef DefaultPrediction* calculate_default_prediction(self, LabelMatrix label_matrix)


cdef class LabelWiseRuleEvaluation:

    # Attributes:

    cdef float64 l2_regularization_weight

    # Functions:

    cdef void calculate_label_wise_prediction(self, const intp[::1] label_indices,
                                              const float64[::1] total_sums_of_gradients,
                                              float64[::1] sums_of_gradients, const float64[::1] total_sums_of_hessians,
                                              float64[::1] sums_of_hessians, bint uncovered,
                                              LabelWisePrediction* prediction)
