from boomer.common._arrays cimport uint8, intp, float64
from boomer.common.statistics cimport LabelMatrix
from boomer.common.rule_evaluation cimport DefaultPrediction, LabelWisePrediction, DefaultRuleEvaluation
from boomer.seco.heuristics cimport Heuristic, AbstractHeuristic

from libcpp cimport bool


cdef extern from "cpp/label_wise_rule_evaluation.h" namespace "rule_evaluation":

    cdef cppclass CppLabelWiseRuleEvaluation:

        # Constructors:

        CppLabelWiseRuleEvaluation(AbstractHeuristic* heuristic);

        # Functions:

        void calculateLabelWisePrediction(const intp* labelIndices, const uint8* minorityLabels,
                                          const float64* confusionMatricesTotal, const float64* confusionMatricesSubset,
                                          const float64* confusionMatricesCovered, bool uncovered,
                                          LabelWisePrediction* prediction) nogil


cdef class LabelWiseDefaultRuleEvaluation(DefaultRuleEvaluation):

    # Functions:

    cdef DefaultPrediction* calculate_default_prediction(self, LabelMatrix label_matrix)


cdef class LabelWiseRuleEvaluation:

    # Attributes:

    cdef CppLabelWiseRuleEvaluation* rule_evaluation

    # Functions:

    cdef void calculate_label_wise_prediction(self, const intp[::1] label_indices, const uint8[::1] minority_labels,
                                              const float64[:, ::1] confusion_matrices_total,
                                              const float64[:, ::1] confusion_matrices_subset,
                                              float64[:, ::1] confusion_matrices_covered, bint uncovered,
                                              LabelWisePrediction* prediction)
