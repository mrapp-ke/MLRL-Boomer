"""
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides classes that allow to calculate the predictions of rules, as well as corresponding quality scores.
"""


cdef class DefaultRuleEvaluation:
    """
    A base class for all classes that allow to calculate the predictions of a default rule.
    """

    cdef DefaultPrediction* calculate_default_prediction(self, LabelMatrix label_matrix) nogil:
        """
        Calculates the scores to be predicted by a default rule based on the ground truth label matrix.

        :param label_matrix:    A `LabelMatrix` that provides random access to the labels of the training examples
        :return:                A pointer to an object of type `DefaultPrediction`, representing the predictions of the
                                default rule
        """
        cdef AbstractDefaultRuleEvaluation* default_rule_evaluation = self.default_rule_evaluation
        return default_rule_evaluation.calculateDefaultPrediction(label_matrix.label_matrix)
