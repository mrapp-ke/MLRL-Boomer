from boomer.common._arrays cimport uint8, intp, float64
from boomer.common._predictions cimport LabelWisePredictionCandidate
from boomer.seco.heuristics cimport AbstractHeuristic

from libcpp cimport bool
from libcpp.memory cimport shared_ptr


cdef extern from "cpp/label_wise_rule_evaluation.h" namespace "seco" nogil:

    cdef cppclass AbstractLabelWiseRuleEvaluation:

        # Functions:

        void calculateLabelWisePrediction(const intp* labelIndices, const uint8* minorityLabels,
                                          const float64* confusionMatricesTotal, const float64* confusionMatricesSubset,
                                          const float64* confusionMatricesCovered, bool uncovered,
                                          LabelWisePredictionCandidate* prediction) except +

    cdef cppclass HeuristicLabelWiseRuleEvaluationImpl(AbstractLabelWiseRuleEvaluation):

        # Constructors:

        HeuristicLabelWiseRuleEvaluationImpl(AbstractHeuristic* heuristic, bool predictMajority) except +

        # Functions:

        void calculateLabelWisePrediction(const intp* labelIndices, const uint8* minorityLabels,
                                          const float64* confusionMatricesTotal, const float64* confusionMatricesSubset,
                                          const float64* confusionMatricesCovered, bool uncovered,
                                          LabelWisePredictionCandidate* prediction) except +


cdef class LabelWiseRuleEvaluation:

    # Attributes:

    cdef shared_ptr[AbstractLabelWiseRuleEvaluation] rule_evaluation_ptr


cdef class HeuristicLabelWiseRuleEvaluation(LabelWiseRuleEvaluation):
    pass
