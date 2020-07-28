"""
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides classes for calculating the predictions of default rules, as well as Cython wrappers for C++ classes that allow
to calculate the predictions of rules.
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
    A wrapper for the C++ class `LabelWiseRuleEvaluationImpl`.
    """

    def __cinit__(self, Heuristic heuristic):
        """
        :param heuristic: The heuristic that should be used
        """
        self.rule_evaluation = new LabelWiseRuleEvaluationImpl(heuristic.heuristic)

    def __dealloc__(self):
        del self.rule_evaluation
