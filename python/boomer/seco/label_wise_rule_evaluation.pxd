from boomer.common._arrays cimport uint8, intp, float64
from boomer.common.statistics cimport LabelMatrix
from boomer.common.rule_evaluation cimport DefaultPrediction, LabelWisePrediction, DefaultRuleEvaluation
from boomer.seco.heuristics cimport Heuristic, AbstractHeuristic


cdef class LabelWiseDefaultRuleEvaluation(DefaultRuleEvaluation):

    # Functions:

    cdef DefaultPrediction* calculate_default_prediction(self, LabelMatrix label_matrix)


cdef class LabelWiseRuleEvaluation:

    # Attributes:

    cdef AbstractHeuristic* heuristic_function

    # Functions:

    cdef void calculate_label_wise_prediction(self, const intp[::1] label_indices, const uint8[::1] minority_labels,
                                              const float64[:, ::1] confusion_matrices_total,
                                              const float64[:, ::1] confusion_matrices_subset,
                                              float64[:, ::1] confusion_matrices_covered, bint uncovered,
                                              LabelWisePrediction* prediction) nogil
