"""
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides Cython wrappers for C++ classes that allow to calculate the predictions of rules, as well as corresponding
quality scores.
"""
from boomer.boosting.label_wise_losses cimport LabelWiseLoss


cdef class LabelWiseDefaultRuleEvaluation(DefaultRuleEvaluation):
    """
    A wrapper for the C++ class `LabelWiseDefaultRuleEvaluationImpl`.
    """

    def __cinit__(self, LabelWiseLoss loss_function, float64 l2_regularization_weight):
        """
        :param loss_function:               The loss function to be minimized
        :param l2_regularization_weight:    The weight of the L2 regularization that is applied for calculating the
                                            scores to be predicted by the default rule
        """
        self.default_rule_evaluation = new LabelWiseDefaultRuleEvaluationImpl(loss_function.loss_function,
                                                                              l2_regularization_weight)

    def __dealloc__(self):
        del self.default_rule_evaluation


cdef class LabelWiseRuleEvaluation:
    """
    A wrapper for the C++ class `LabelWiseRuleEvaluationImpl`.
    """

    def __cinit__(self, float64 l2_regularization_weight):
        """
        :param l2_regularization_weight: The weight of the L2 regularization that is applied for calculating the scores
                                         to be predicted by rules
        """
        self.rule_evaluation = new LabelWiseRuleEvaluationImpl(l2_regularization_weight)

    def __dealloc__(self):
        del self.rule_evaluation
