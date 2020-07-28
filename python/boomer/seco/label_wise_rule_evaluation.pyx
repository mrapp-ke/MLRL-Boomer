"""
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides classes that allow to calculate the predictions of rules, as well as corresponding quality scores, such that
the optimize a heuristic that is applied using label-wise averaging.
"""
from boomer.common._arrays cimport get_index
from boomer.seco.heuristics cimport Element

from libc.stdlib cimport malloc


cdef class LabelWiseDefaultRuleEvaluation(DefaultRuleEvaluation):
    """
    Allows to calculate the predictions of a default rule such that it optimizes a heuristic that is applied using
    label-wise averaging.
    """

    cdef DefaultPrediction* calculate_default_prediction(self, LabelMatrix label_matrix):
        # The number of examples
        cdef intp num_examples = label_matrix.num_examples
        # The number of labels
        cdef intp num_labels = label_matrix.num_labels
        # The number of positive examples that must be exceeded for the default rule to predict a label as positive
        cdef float64 threshold = num_examples / 2.0
        # An array that stores the scores that are predicted by the default rule
        cdef float64* predicted_scores = <float64*>malloc(num_labels * sizeof(float64))
        # Temporary variables
        cdef intp num_positive_labels
        cdef uint8 true_label
        cdef intp c, r

        for c in range(num_labels):
            num_positive_labels = 0

            for r in range(num_examples):
                true_label = label_matrix.get_label(r, c)

                if true_label:
                    num_positive_labels += 1

            predicted_scores[c] = (num_positive_labels > threshold)

        return new DefaultPrediction(num_labels, predicted_scores)


cdef class LabelWiseRuleEvaluation:
    """
    Allows to calculate the predictions of rules, as well as corresponding quality scores, such that they optimize a
    heuristic that is applied using label-wise averaging.
    """

    def __cinit__(self, Heuristic heuristic):
        """
        :param heuristic: The heuristic that should be used
        """
        cdef HeuristicFunction* heuristic_function = heuristic.heuristic_function
        self.rule_evaluation = new CppLabelWiseRuleEvaluation(heuristic_function)

    def __dealloc__(self):
        del self.rule_evaluation

    cdef void calculate_label_wise_prediction(self, const intp[::1] label_indices, const uint8[::1] minority_labels,
                                              const float64[::1, :] confusion_matrices_total,
                                              const float64[::1, :] confusion_matrices_subset,
                                              float64[::1, :] confusion_matrices_covered, bint uncovered,
                                              LabelWisePrediction* prediction):
        """
        Calculates the scores to be predicted by a rule, as well as corresponding quality scores, based on confusion
        matrices. The predicted scores and quality scores are stored in a given object of type `LabelWisePrediction`.

        :param label_indices:               An array of dtype `intp`, shape `prediction.numPredictions_)`, representing
                                            the indices of the labels for which the rule should predict or None, if the
                                            rule should predict for all labels
        :param minority_labels:             An array of dtype `uint8`, shape `(num_labels)`, indicating whether the rule
                                            should predict individual labels as positive (1) or negative (0)
        :param confusion_matrices_total:    A matrix of dtype `float64`, shape `(num_labels, 4)`, storing a confusion
                                            matrix, which corresponds to all examples, for each label
        :param confusion_matrices_subset:   A matrix of dtype `float64`, shape `(num_labels, 4)`, storing a confusion
                                            matrix, which corresponds to all examples covered by the previous refinement
                                            of the rule, for each label
        :param confusion_matrices_covered:  A matrix of dtype `float64`, shape `(prediction.numPredictions_)`, storing a
                                            confusion matrix, which corresponds to all examples covered by the rule, for
                                            each label
        :param uncovered:                   0, if the confusion matrices in `confusion_matrices_covered` correspond to
                                            the examples that are covered by rule, 1, if they correspond to the examples
                                            that are not covered by the rule
        :param prediction:                  A pointer to an object of type `LabelWisePrediction` that should be used to
                                            store the predicted scores and quality scores
        """
        cdef CppLabelWiseRuleEvaluation* rule_evaluation = self.rule_evaluation
        rule_evaluation.calculateLabelWisePrediction(&label_indices[0], &minority_labels[0],
                                                     &confusion_matrices_total[0][0], &confusion_matrices_subset[0][0],
                                                     &confusion_matrices_covered[0][0], uncovered, prediction)
