"""
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides classes that allow to calculate predictions, as well as corresponding quality scores, that minimize loss
functions that are applied label-wise.
"""
from boomer.common._arrays cimport get_index
from boomer.boosting.differentiable_losses cimport _l2_norm_pow

from libc.stdlib cimport malloc
from libc.math cimport pow

from libcpp.pair cimport pair


cdef class LabelWiseDefaultRuleEvaluation:
    """
    Allows to calculate the predictions of the default rule such that they minimize a loss function that is applied
    label-wise.
    """

    def __cinit__(self, LabelWiseLossFunction loss_function, float64 l2_regularization_weight):
        """
        :param loss_function:               The loss function to be minimized
        :param l2_regularization_weight:    The weight of the L2 regularization that is applied for calculating the
                                            scores to be predicted by the default rule
        """
        self.loss_function = loss_function
        self.l2_regularization_weight = l2_regularization_weight

    cdef DefaultPrediction* calculate_default_prediction(self, LabelMatrix label_matrix):
        """
        Calculates the scores to be predicted by the default rule.

        :param label_matrix:    A `LabelMatrix` that provides random access to the labels of the training examples
        :return:                A pointer to an object of type `DefaultPrediction`, representing the predictions of the
                                default rule
        """
        # Class members
        cdef LabelWiseLossFunction loss_function = self.loss_function
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
    Allows to calculate the predictions of rules, as well as corresponding quality scores, such that they minimize a
    loss function that is applied label-wise.
    """

    def __cinit__(self, float64 l2_regularization_weight):
        """
        :param l2_regularization_weight: The weight of the L2 regularization that is applied for calculating the scores
                                         to be predicted by rules
        """
        self.l2_regularization_weight = l2_regularization_weight

    cdef void calculate_label_wise_prediction(self, const intp[::1] label_indices,
                                              const float64[::1] total_sums_of_gradients,
                                              const float64[::1] sums_of_gradients,
                                              const float64[::1] total_sums_of_hessians,
                                              const float64[::1] sums_of_hessians, bint uncovered,
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
        :param total_sums_of_gradients: An array of dtype `float64`, shape `(prediction.numPredictions), representing
                                        the total sums of gradients for individual labels
        :param sums_of_gradients:       An array of dtype `float64`, shape `(prediction.numPredictions_)`, representing
                                        the sums of gradients for individual labels
        :param total_sums_of_hessians:  An array of dtype `float64`, shape `(prediction.numPredictions_)`, representing
                                        the total sums of Hessians for individual labels
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
        # Class members
        cdef float64 l2_regularization_weight = self.l2_regularization_weight
        # The number of labels to predict for
        cdef intp num_labels = sums_of_gradients.shape[0]
        # The array that should be used to store the predicted scores
        cdef float64* predicted_scores = prediction.predictedScores_
        # The array that should be used to store the quality scores
        cdef float64* quality_scores = prediction.qualityScores_
        # The overall quality score, i.e., the sum of the quality scores for each label plus the L2 regularization term
        cdef float64 overall_quality_score = 0
        # Temporary variables
        cdef float64 sum_of_gradients, sum_of_hessians, score, score_pow
        cdef intp c, l

        # For each label, calculate a score to be predicted, as well as a corresponding quality score...
        for c in range(num_labels):
            sum_of_gradients = sums_of_gradients[c]
            sum_of_hessians = sums_of_hessians[c]

            if uncovered:
                l = get_index(c, label_indices)
                sum_of_gradients = total_sums_of_gradients[l] - sum_of_gradients
                sum_of_hessians = total_sums_of_hessians[l] - sum_of_hessians

            # Calculate the score to be predicted for the current label...
            score = sum_of_hessians + l2_regularization_weight
            score = -sum_of_gradients / score if score != 0 else 0
            predicted_scores[c] = score

            # Calculate the quality score for the current label...
            score_pow = pow(score, 2)
            score = (sum_of_gradients * score) + (0.5 * score_pow * sum_of_hessians)
            quality_scores[c] = score + (0.5 * l2_regularization_weight * score_pow)
            overall_quality_score += score

        # Add the L2 regularization term to the overall quality score...
        overall_quality_score += 0.5 * l2_regularization_weight * _l2_norm_pow(predicted_scores, num_labels)
        prediction.overallQualityScore_ = overall_quality_score
