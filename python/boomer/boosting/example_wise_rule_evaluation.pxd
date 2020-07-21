from boomer.common._arrays cimport float64
from boomer.common.statistics cimport LabelMatrix
from boomer.common.rule_evaluation cimport DefaultPrediction, DefaultRuleEvaluation
from boomer.boosting.losses cimport ExampleWiseLossFunction


cdef class ExampleWiseDefaultRuleEvaluation(DefaultRuleEvaluation):

    # Attributes:

    cdef ExampleWiseLossFunction loss_function

    cdef float64 l2_regularization_weight

    # Functions:

    cdef DefaultPrediction* calculate_default_prediction(self, LabelMatrix label_matrix)
