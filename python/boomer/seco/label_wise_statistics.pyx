"""
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides classes that allow to store the elements of confusion matrices that are computed independently for each label.
"""
from boomer.common._arrays cimport array_uint8, c_matrix_float64, get_index
from boomer.seco.heuristics cimport ConfusionMatrixElement


cdef class LabelWiseRefinementSearch(RefinementSearch):
    """
    A wrapper for the C++ class `LabelWiseRefinementSearchImpl`.
    """

    def __cinit__(self, LabelWiseRuleEvaluation rule_evaluation, const intp[::1] label_indices,
                  LabelMatrix label_matrix, const float64[:, ::1] uncovered_labels, const uint8[::1] minority_labels,
                  const float64[:, ::1] confusion_matrices_total, const float64[:, ::1] confusion_matrices_subset):
        """
        :param rule_evaluation:             The `LabelWiseRuleEvaluation` to be used for calculating the predictions, as
                                            well as corresponding quality scores, of rules
        :param label_indices:               An array of dtype int, shape `(num_considered_labels)`, representing the
                                            indices of the labels that should be considered by the search or None, if
                                            all labels should be considered
        :param label_matrix:                A `LabelMatrix` that provides random access to the labels of the training
                                            examples
        :param uncovered_labels:            An array of dtype float, shape `(num_examples, num_labels)`, indicating
                                            which each examples and labels remain to be covered
        :param minority_labels:             An array of dtype `uint8`, shape `(num_labels)`, indicating whether rules
                                            should predict individual labels as positive (1) or negative (0)
        :param confusion_matrices_total:    A matrix of dtype float, shape `(num_labels, 4)`, storing a confusion matrix
                                            that corresponds to all examples, for each label
        :param confusion_matrices_subset:   A matrix of dtype float, shape `(num_labels, 4)`, storing a confusion matrix
                                            that corresponds to all examples that are covered by the previous refinement
                                            of a rule, for each label
        """
        cdef intp num_predictions = minority_labels.shape[0] if label_indices is None else label_indices.shape[0]
        cdef const intp* label_indices_ptr = <const intp*>NULL if label_indices is None else &label_indices[0]
        self.refinement_search = new LabelWiseRefinementSearchImpl(rule_evaluation.rule_evaluation, num_predictions,
                                                                   &label_indices[0], label_matrix.label_matrix,
                                                                   &uncovered_labels[0, 0], &minority_labels[0],
                                                                   &confusion_matrices_total[0, 0],
                                                                   &confusion_matrices_subset[0, 0])

    def __dealloc__(self):
        del self.refinement_search

    cdef void update_search(self, intp statistic_index, uint32 weight):
        cdef AbstractRefinementSearch* refinement_search = self.refinement_search
        refinement_search.updateSearch(statistic_index, weight)

    cdef void reset_search(self):
        cdef AbstractRefinementSearch* refinement_search = self.refinement_search
        refinement_search.resetSearch()

    cdef LabelWisePrediction* calculate_label_wise_prediction(self, bint uncovered, bint accumulated) nogil:
        cdef AbstractRefinementSearch* refinement_search = self.refinement_search
        return refinement_search.calculateLabelWisePrediction(uncovered, accumulated)

    cdef Prediction* calculate_example_wise_prediction(self, bint uncovered, bint accumulated) nogil:
        cdef AbstractRefinementSearch* refinement_search = self.refinement_search
        return refinement_search.calculateExampleWisePrediction(uncovered, accumulated)


cdef class LabelWiseStatistics(CoverageStatistics):
    """
    Allows to store the elements of confusion matrices that are computed independently for each label.
    """

    def __cinit__(self, LabelWiseRuleEvaluation rule_evaluation):
        """
        :param rule_evaluation: The `LabelWiseRuleEvaluation` to be used for calculating the predictions, as well as
                                corresponding quality scores, of rules
        """
        self.rule_evaluation = rule_evaluation

    cdef void apply_default_prediction(self, LabelMatrix label_matrix, DefaultPrediction* default_prediction):
        # The number of examples
        cdef intp num_examples = label_matrix.num_examples
        # The number of labels
        cdef intp num_labels = label_matrix.num_labels
        # A matrix that stores whether individual examples and labels are still uncovered
        cdef float64[:, ::1] uncovered_labels = c_matrix_float64(num_examples, num_labels)
        # The total number of uncovered examples and labels
        cdef float64 sum_uncovered_labels = 0
        # An array that stores whether rules should predict individual labels as positive (1) or negative (0)
        cdef uint8[::1] minority_labels = array_uint8(num_labels)
        # A matrix that stores a confusion matrix, which corresponds to all examples, for each label
        cdef float64[:, ::1] confusion_matrices_total = c_matrix_float64(num_labels, 4)
        # A matrix that stores a confusion matrix, which corresponds to the examples covered by the previous refinement
        # of a rule, for each label
        cdef float64[:, ::1] confusion_matrices_subset = c_matrix_float64(num_labels, 4)
        # An array that stores the predictions of the default rule or NULL, if no default rule is used
        cdef float64* predicted_scores = default_prediction.predictedScores_ if default_prediction != NULL else NULL
        # Temporary variables
        cdef uint8 predicted_label, true_label
        cdef intp c, r

        for c in range(num_labels):
            predicted_label = <uint8>predicted_scores[c] if predicted_scores != NULL else 0

            # Rules should predict the opposite of the default rule...
            minority_labels[c] = not predicted_label

            for r in range(num_examples):
                true_label = label_matrix.get_label(r, c)

                # Increment the total number of uncovered labels, if the default rule's prediction for the current
                # example and label is incorrect...
                if true_label != predicted_label:
                    sum_uncovered_labels += 1

                # Mark the current example and label as uncovered...
                uncovered_labels[r, c] = 1

        # Store class members...
        self.label_matrix = label_matrix
        self.uncovered_labels = uncovered_labels
        self.sum_uncovered_labels = sum_uncovered_labels
        self.minority_labels = minority_labels
        self.confusion_matrices_total = confusion_matrices_total
        self.confusion_matrices_subset = confusion_matrices_subset

    cdef void reset_sampled_statistics(self):
        # Class members
        cdef float64[:, ::1] confusion_matrices_total = self.confusion_matrices_total
        cdef float64[:, ::1] confusion_matrices_subset = self.confusion_matrices_subset
        # The number of examples
        cdef intp num_examples = confusion_matrices_total.shape[0]
        # The number of labels
        cdef intp num_labels = confusion_matrices_total.shape[1]
        # Temporary variables
        cdef intp r, c

        # Reset confusion matrices to 0...
        for r in range(num_examples):
            for c in range(num_labels):
                confusion_matrices_total[r, c] = 0
                confusion_matrices_subset[r,c] = 0

    cdef void add_sampled_statistic(self, intp statistic_index, uint32 weight):
        # Class members
        cdef LabelMatrix label_matrix = self.label_matrix
        cdef float64[:, ::1] uncovered_labels = self.uncovered_labels
        cdef uint8[::1] minority_labels = self.minority_labels
        cdef float64[:, ::1] confusion_matrices_total = self.confusion_matrices_total
        cdef float64[:, ::1] confusion_matrices_subset = self.confusion_matrices_subset
        # The number of labels
        cdef intp num_labels = minority_labels.shape[0]
        # Temporary variables
        cdef uint8 uncovered, true_label, predicted_label
        cdef intp c, element

        for c in range(num_labels):
            uncovered = <uint8>uncovered_labels[statistic_index, c]

            # Only uncovered labels must be considered...
            if uncovered:
                # Add the current example and label to the confusion matrix for the current label...
                true_label = label_matrix.get_label(statistic_index, c)
                predicted_label = minority_labels[c]
                element = __get_confusion_matrix_element(true_label, predicted_label)
                confusion_matrices_total[c, element] += weight
                confusion_matrices_subset[c, element] += weight

    cdef void reset_covered_statistics(self):
        # Class members
        cdef float64[:, ::1] confusion_matrices_subset = self.confusion_matrices_subset

        # Reset confusion matrices to 0...
        confusion_matrices_subset[:, :] = 0

    cdef void update_covered_statistic(self, intp statistic_index, uint32 weight, bint remove):
        # Class members
        cdef LabelMatrix label_matrix = self.label_matrix
        cdef float64[:, ::1] uncovered_labels = self.uncovered_labels
        cdef uint8[::1] minority_labels = self.minority_labels
        cdef float64[:, ::1] confusion_matrices_subset = self.confusion_matrices_subset
        # The number of labels
        cdef intp num_labels = minority_labels.shape[0]
        # The given weight multiplied by 1 or -1, depending on the argument `remove`
        cdef float64 signed_weight = -<float64>weight if remove else weight
        # Temporary variables
        cdef uint8 uncovered, true_label, predicted_label
        cdef intp c, element

        for c in range(num_labels):
            uncovered = <uint8>uncovered_labels[statistic_index, c]

            # Only uncovered labels must be considered...
            if uncovered:
                # Add the current example and label to the confusion matrix for the current label...
                true_label = label_matrix.get_label(statistic_index, c)
                predicted_label = minority_labels[c]
                element = __get_confusion_matrix_element(true_label, predicted_label)
                confusion_matrices_subset[c, element] += signed_weight

    cdef AbstractRefinementSearch* begin_search(self, intp[::1] label_indices):
        # Class members
        cdef LabelWiseRuleEvaluation rule_evaluation = self.rule_evaluation
        cdef LabelMatrix label_matrix = self.label_matrix
        cdef float64[:, ::1] uncovered_labels = self.uncovered_labels
        cdef uint8[::1] minority_labels = self.minority_labels
        cdef float64[:, ::1] confusion_matrices_total = self.confusion_matrices_total
        cdef float64[:, ::1] confusion_matrices_subset = self.confusion_matrices_subset

        # Instantiate and return a new object of the class `LabelWiseRefinementSearchImpl`...
        cdef intp num_predictions = minority_labels.shape[0] if label_indices is None else label_indices.shape[0]
        cdef const intp* label_indices_ptr = <const intp*>NULL if label_indices is None else &label_indices[0]
        return new LabelWiseRefinementSearchImpl(rule_evaluation.rule_evaluation, num_predictions, &label_indices[0],
                                                 label_matrix.label_matrix, &uncovered_labels[0, 0],
                                                 &minority_labels[0], &confusion_matrices_total[0, 0],
                                                 &confusion_matrices_subset[0, 0])

    cdef void apply_prediction(self, intp statistic_index, intp[::1] label_indices, HeadCandidate* head):
        # Class members
        cdef LabelMatrix label_matrix = self.label_matrix
        cdef float64[:, ::1] uncovered_labels = self.uncovered_labels
        cdef float64 sum_uncovered_labels = self.sum_uncovered_labels
        cdef uint8[::1] minority_labels = self.minority_labels
        # The number of predicted labels
        cdef intp num_predictions = head.numPredictions_
        # Temporary variables
        cdef uint8 uncovered, true_label, predicted_label
        cdef intp c, l

        # Only the labels that are predicted by the new rule must be considered...
        for c in range(num_predictions):
            l = get_index(c, label_indices)
            uncovered = <uint8>uncovered_labels[statistic_index, l]

            if uncovered:
                true_label = label_matrix.get_label(statistic_index, l)
                predicted_label = minority_labels[l]

                # Decrement the total number of uncovered labels, if the default rule's prediction for the current
                # example and label is incorrect...
                if predicted_label == true_label:
                    sum_uncovered_labels -= 1

                # Mark the label as covered...
                uncovered_labels[statistic_index, l] = 0

        # Update the total number of uncovered labels...
        self.sum_uncovered_labels = sum_uncovered_labels


cdef inline intp __get_confusion_matrix_element(uint8 true_label, uint8 predicted_label):
    """
    Returns the element of a confusion matrix, a label corresponds to depending on the ground truth and a prediction.

    :param true_label:      The true label
    :param predicted_label: The predicted label
    :return:                The confusion matrix element
    """
    if true_label:
        return <intp>ConfusionMatrixElement.RP if predicted_label else <intp>ConfusionMatrixElement.RN
    else:
        return <intp>ConfusionMatrixElement.IP if predicted_label else <intp>ConfusionMatrixElement.IN
