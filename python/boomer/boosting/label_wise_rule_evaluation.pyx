"""
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides classes that allow to calculate the predictions of default rules, as well as Cython wrappers for C++ classes
that allow to calculate the predictions of rules.
"""
from boomer.common._arrays cimport get_index
from boomer.boosting._math cimport l2_norm_pow

from libc.stdlib cimport malloc
from libc.math cimport pow

from libcpp.pair cimport pair


cdef class LabelWiseDefaultRuleEvaluation(DefaultRuleEvaluation):
    """
    Allows to calculate the predictions of a default rule such that it minimizes a loss function that is applied
    label-wise.
    """

    def __cinit__(self, LabelWiseLoss loss_function, float64 l2_regularization_weight):
        """
        :param loss_function:               The loss function to be minimized
        :param l2_regularization_weight:    The weight of the L2 regularization that is applied for calculating the
                                            scores to be predicted by the default rule
        """
        self.loss_function = loss_function
        self.l2_regularization_weight = l2_regularization_weight

    cdef DefaultPrediction* calculate_default_prediction(self, LabelMatrix label_matrix):
        # Class members
        cdef LabelWiseLoss loss_function = self.loss_function
        cdef float64 l2_regularization_weight = self.l2_regularization_weight
        # The number of examples
        cdef intp num_examples = label_matrix.num_examples
        # The number of labels
        cdef intp num_labels = label_matrix.num_labels
        # An array that stores the scores that are predicted by the default rule
        cdef float64* predicted_scores = <float64*>malloc(num_labels * sizeof(float64))
        # Temporary variables
        cdef pair[float64, float64] gradient_and_hessian
        cdef float64 gradient, sum_of_gradients, hessian, sum_of_hessians, predicted_score
        cdef intp c, r

        for c in range(num_labels):
            sum_of_gradients = 0
            sum_of_hessians = 0

            for r in range(num_examples):
                # Calculate the gradient and Hessian for the current example and label...
                gradient_and_hessian = loss_function.calculate_gradient_and_hessian(label_matrix, r, c, 0)
                gradient = gradient_and_hessian.first
                sum_of_gradients += gradient
                hessian = gradient_and_hessian.second
                sum_of_hessians += hessian

            # Calculate the score to be predicted by the default rule for the current label...
            predicted_score = -sum_of_gradients / (sum_of_hessians + l2_regularization_weight)
            predicted_scores[c] = predicted_score

        return new DefaultPrediction(num_labels, predicted_scores)


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

    cdef void calculate_label_wise_prediction(self, const intp[::1] label_indices,
                                              const float64[::1] total_sums_of_gradients,
                                              float64[::1] sums_of_gradients, const float64[::1] total_sums_of_hessians,
                                              float64[::1] sums_of_hessians, bint uncovered,
                                              LabelWisePrediction* prediction):
        """
        Calculates the scores to be predicted by a rule, as well as corresponding quality scores, based on the
        label-wise sums of gradients and Hessians that are covered by the rule. The predicted scores and quality scores
        are stored in a given object of type `LabelWisePrediction`.

        If the argument `uncovered` is 1, the rule is considered to cover the difference between the sums of gradients
        and Hessians that are stored in the arrays `total_sums_of_gradients` and `sums_of_gradients` and
        `total_sums_of_hessians` and `sums_of_hessians`, respectively.

        :param label_indices:           An array of dtype `intp`, shape `prediction.numPredictions_)`, representing the
                                        indices of the labels for which the rule should predict or None, if the rule
                                        should predict for all labels
        :param total_sums_of_gradients: An array of dtype `float64`, shape `(num_labels), representing the total sums of
                                        gradients for individual labels
        :param sums_of_gradients:       An array of dtype `float64`, shape `(prediction.numPredictions_)`, representing
                                        the sums of gradients for individual labels
        :param total_sums_of_hessians:  An array of dtype `float64`, shape `(num_labels)`, representing the total sums
                                        of Hessians for individual labels
        :param sums_of_hessians:        An array of dtype `float64`, shape `(prediction.numPredictions_)`, representing
                                        the sums of Hessians for individual labels
        :param uncovered:               0, if the rule covers the sums of gradient and Hessians that are stored in the
                                        array `sums_of_gradients` and `sums_of_hessians`, 1, if the rule covers the
                                        difference between the sums of gradients and Hessians that are stored in the
                                        arrays `total_sums_of_gradients` and `sums_of_gradients` and
                                        `total_sums_of_hessians` and `sums_of_hessians`, respectively.
        :param prediction:              A pointer to an object of type `LabelWisePrediction` that should be used to
                                        store the predicted scores and quality scores
        """
        cdef LabelWiseRuleEvaluationImpl* rule_evaluation = self.rule_evaluation
        cdef const intp* label_indices_ptr = <const intp*>NULL if label_indices is None else &label_indices[0]
        rule_evaluation.calculateLabelWisePrediction(label_indices_ptr, &total_sums_of_gradients[0],
                                          &sums_of_gradients[0], &total_sums_of_hessians[0], &sums_of_hessians[0],
                                          uncovered, prediction)
