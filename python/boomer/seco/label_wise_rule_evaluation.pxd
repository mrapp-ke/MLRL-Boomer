from boomer.common.statistics cimport LabelMatrix
from boomer.common.rule_evaluation cimport DefaultPrediction, DefaultRuleEvaluation
from boomer.seco.heuristics cimport Heuristic


cdef class LabelWiseDefaultRuleEvaluation(DefaultRuleEvaluation):

    # Attributes:

    cdef Heuristic heuristic

    # Functions:

    cdef DefaultPrediction* calculate_default_prediction(self, LabelMatrix label_matrix)
