"""
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides classes that allow to calculate the predictions of rules, as well as corresponding quality scores, such that
they minimize a loss function that is applied example-wise.
"""
from boomer.boosting._blas cimport init_blas
from boomer.boosting._lapack cimport init_lapack


cdef class ExampleWiseDefaultRuleEvaluation(DefaultRuleEvaluation):
    """
    A wrapper for the C++ class `ExampleWiseDefaultRuleEvaluationImpl`.
    """

    def __cinit__(self, ExampleWiseLoss loss_function, float64 l2_regularization_weight):
        """
        :param loss_function:               The loss function to be minimized
        :param l2_regularization_weight:    The weight of the L2 regularization that is applied for calculating the
                                            scores to be predicted by the default rule
        """
        cdef Lapack* lapack = init_lapack()
        self.default_rule_evaluation = new ExampleWiseDefaultRuleEvaluationImpl(loss_function.loss_function,
                                                                                l2_regularization_weight, lapack)

    def __dealloc__(self):
        del self.default_rule_evaluation

    cdef DefaultPrediction* calculate_default_prediction(self, LabelMatrix label_matrix):
        cdef AbstractDefaultRuleEvaluation* default_rule_evaluation = self.default_rule_evaluation
        return default_rule_evaluation.calculateDefaultPrediction(label_matrix.label_matrix)


cdef class ExampleWiseRuleEvaluation:
    """
    A wrapper for the C++ class `ExampleWiseRuleEvaluationImpl`.
    """

    def __cinit__(self, float64 l2_regularization_weight):
        """
        :param l2_regularization_weight: The weight of the L2 regularization that is applied for calculating the scores
                                         to be predicted by rules
        """
        cdef Blas* blas = init_blas()
        cdef Lapack* lapack = init_lapack()
        self.rule_evaluation = new ExampleWiseRuleEvaluationImpl(l2_regularization_weight, blas, lapack)

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

        :param label_indices:           An array of dtype `intp`, shape `(num_gradients)`, representing the indices of
                                        the labels for which the rule should predict or None, if the rule should predict
                                        for all labels
        :param total_sums_of_gradients: An array of dtype `float64`, shape `(num_gradients), representing the total sums
                                        of gradients for individual labels
        :param sums_of_gradients:       An array of dtype `float64`, shape `(num_gradients)`, representing the sums of
                                        gradients for individual labels
        :param total_sums_of_hessians:  An array of dtype `float64`, shape
                                        `((num_gradients + (num_gradients + 1)) // 2)`, representing the total sums of
                                        Hessians for individual labels
        :param sums_of_hessians:        An array of dtype `float64`, shape
                                        `((num_gradients + (num_gradients + 1)) // 2)`, representing the sums of
                                        Hessians for individual labels
        :param uncovered:               0, if the rule covers the sums of gradient and Hessians that are stored in the
                                        array `sums_of_gradients` and `sums_of_hessians`, 1, if the rule covers the
                                        difference between the sums of gradients and Hessians that are stored in the
                                        arrays `total_sums_of_gradients` and `sums_of_gradients` and
                                        `total_sums_of_hessians` and `sums_of_hessians`, respectively.
        :param prediction:              A pointer to an object of type `LabelWisePrediction` that should be used to
                                        store the predicted scores and quality scores
        """
        cdef ExampleWiseRuleEvaluationImpl* rule_evaluation = self.rule_evaluation
        cdef const intp* label_indices_ptr = <const intp*>NULL if label_indices is None else &label_indices[0]
        rule_evaluation.calculateLabelWisePrediction(label_indices_ptr, &total_sums_of_gradients[0],
                                                     &sums_of_gradients[0], &total_sums_of_hessians[0],
                                                     &sums_of_hessians[0], uncovered, prediction)

    cdef void calculate_example_wise_prediction(self, const intp[::1] label_indices,
                                                const float64[::1] total_sums_of_gradients,
                                                float64[::1] sums_of_gradients,
                                                const float64[::1] total_sums_of_hessians,
                                                float64[::1] sums_of_hessians, bint uncovered,
                                                Prediction* prediction):
        """
        Calculates the scores to be predicted by a rule, as well as an overall quality score, based on the sums of
        gradients and Hessians that are covered by the rule. The predicted scores and quality scores are stored in a
        given object of type `Prediction`.

        If the argument `uncovered` is 1, the rule is considered to cover the difference between the sums of gradients
        and Hessians that are stored in the arrays `total_sums_of_gradients` and `sums_of_gradients` and
        `total_sums_of_hessians` and `sums_of_hessians`, respectively.

        :param label_indices:           An array of dtype `intp`, shape `(num_gradients)`, representing the indices of
                                        the labels for which the rule should predict or None, if the rule should predict
                                        for all labels
        :param total_sums_of_gradients: An array of dtype `float64`, shape `(num_gradients), representing the total sums
                                        of gradients for individual labels
        :param sums_of_gradients:       An array of dtype `float64`, shape `(prediction.numPredictions_)`, representing
                                        the sums of gradients for individual labels
        :param total_sums_of_hessians:  An array of dtype `float64`, shape
                                        `((num_gradients + (num_gradients + 1)) // 2)`, representing the total sums of
                                        Hessians for individual labels
        :param sums_of_hessians:        An array of dtype `float64`, shape
                                        `((prediction.numPredictions_ + (prediction.numPredictions_ + 1)) // 2)`,
                                        representing the sums of Hessians for individual labels
        :param uncovered:               0, if the rule covers the sums of gradient and Hessians that are stored in the
                                        array `sums_of_gradients` and `sums_of_hessians`, 1, if the rule covers the
                                        difference between the sums of gradients and Hessians that are stored in the
                                        arrays `total_sums_of_gradients` and `sums_of_gradients` and
                                        `total_sums_of_hessians` and `sums_of_hessians`, respectively.
        :param prediction:              A pointer to an object of type `Prediction` that should be used to store the
                                        predicted scores and quality score
        """
        cdef ExampleWiseRuleEvaluationImpl* rule_evaluation = self.rule_evaluation
        cdef const intp* label_indices_ptr = <const intp*>NULL if label_indices is None else &label_indices[0]
        rule_evaluation.calculateExampleWisePrediction(label_indices_ptr, &total_sums_of_gradients[0],
                                                       &sums_of_gradients[0], &total_sums_of_hessians[0],
                                                       &sums_of_hessians[0], uncovered, prediction)
