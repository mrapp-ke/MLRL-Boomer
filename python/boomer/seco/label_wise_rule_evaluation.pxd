from boomer.common._arrays cimport uint8, intp, float64
from boomer.common.input_data cimport LabelMatrix, AbstractLabelMatrix
from boomer.common.rule_evaluation cimport DefaultPrediction, LabelWisePrediction, DefaultRuleEvaluation, \
    AbstractDefaultRuleEvaluation
from boomer.seco.heuristics cimport Heuristic, AbstractHeuristic

from libcpp cimport bool


cdef extern from "cpp/label_wise_rule_evaluation.h" namespace "rule_evaluation":

    cdef cppclass LabelWiseDefaultRuleEvaluationImpl(AbstractDefaultRuleEvaluation):

        # Functions:

        DefaultPrediction* calculateDefaultPrediction(AbstractLabelMatrix* labelMatrix) nogil


    cdef cppclass LabelWiseRuleEvaluationImpl:

        # Constructors:

        LabelWiseRuleEvaluationImpl(AbstractHeuristic* heuristic)

        # Functions:

        void calculateLabelWisePrediction(const intp* labelIndices, const uint8* minorityLabels,
                                          const float64* confusionMatricesTotal, const float64* confusionMatricesSubset,
                                          const float64* confusionMatricesCovered, bool uncovered,
                                          LabelWisePrediction* prediction) nogil


cdef class LabelWiseDefaultRuleEvaluation(DefaultRuleEvaluation):

    # Attributes:

    cdef AbstractDefaultRuleEvaluation* default_rule_evaluation

    # Functions:

    cdef DefaultPrediction* calculate_default_prediction(self, LabelMatrix label_matrix)


cdef class LabelWiseRuleEvaluation:

    # Attributes:

    cdef LabelWiseRuleEvaluationImpl* rule_evaluation
