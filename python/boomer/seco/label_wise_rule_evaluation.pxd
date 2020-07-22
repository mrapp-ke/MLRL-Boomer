from boomer.common.statistics cimport LabelMatrix
from boomer.common.rule_evaluation cimport DefaultPrediction, DefaultRuleEvaluation


cdef class LabelWiseDefaultRuleEvaluation(DefaultRuleEvaluation):

    # Functions:

    cdef DefaultPrediction* calculate_default_prediction(self, LabelMatrix label_matrix)
