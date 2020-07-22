"""
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides classes that allow to store the elements of confusion matrices that are computed independently for each label.
"""
from boomer.common._arrays cimport array_uint8, fortran_matrix_float64, get_index

DEF _IN = 0
DEF _IP = 1
DEF _RN = 2
DEF _RP = 3


cdef class LabelWiseStatistics(CoverageStatistics):
    """
    Allows to store the elements of confusion matrices that are computed independently for each label.
    """

    cdef void apply_default_prediction(self, LabelMatrix label_matrix, DefaultPrediction* default_prediction):
        # The number of examples
        cdef intp num_examples = label_matrix.num_examples
        # The number of labels
        cdef intp num_labels = label_matrix.num_labels
        # A matrix that stores whether individual examples and labels are still uncovered
        cdef float64[::1, :] uncovered_labels = fortran_matrix_float64(num_examples, num_labels)
        # The total number of uncovered examples and labels
        cdef float64 sum_uncovered_labels = 0
        # An array that stores whether rules should predict individual labels as positive (1) or negative (0)
        cdef uint8[::1] minority_labels = array_uint8(num_labels)
        # A matrix that stores a confusion matrix, which corresponds to all examples, for each label
        cdef float64[::1, :] confusion_matrices_default = fortran_matrix_float64(num_labels, 4)
        # A matrix that stores a confusion matrix, which corresponds to the examples covered by a rule, for each label
        cdef float64[::1, :] confusion_matrices_subsample_default = fortran_matrix_float64(num_labels, 4)
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
        self.confusion_matrices_default = confusion_matrices_default
        self.confusion_matrices_subsample_default = confusion_matrices_subsample_default

    cdef void reset_statistics(self):
        # Class members
        cdef LabelMatrix label_matrix = self.label_matrix
        cdef float64[::1, :] uncovered_labels = self.uncovered_labels
        cdef uint8[::1] minority_labels = self.minority_labels
        cdef float64[::1, :] confusion_matrices_default = self.confusion_matrices_default
        cdef float64[::1, :] confusion_matrices_subsample_default = self.confusion_matrices_subsample_default
        # The number of examples
        cdef intp num_examples = label_matrix.num_examples
        # The number of labels
        cdef intp num_labels = label_matrix.num_labels
        # Temporary variables
        cdef uint8 uncovered, true_label, predicted_label
        cdef intp c, r

        for c in range(num_labels):
            predicted_label = minority_labels[c]

            # Reset confusion matrices for the current label to 0...
            for r in range(num_examples):
                confusion_matrices_default[c, r] = 0
                confusion_matrices_subsample_default[c, r] = 0

            for r in range(num_examples):
                uncovered = <uint8>uncovered_labels[r, c]

                # Only uncovered labels must be considered...
                if uncovered:
                    true_label = label_matrix.get_label(r, c)

                    # Add the current example and label to the confusion matrix for the current label...
                    if true_label:
                        if predicted_label:
                            confusion_matrices_default[c, _RP] += 1
                        else:
                            confusion_matrices_default[c, _RN] += 1
                    else:
                        if predicted_label:
                            confusion_matrices_default[c, _IP] += 1
                        else:
                            confusion_matrices_default[c, _IN] += 1

    cdef void update_covered_statistic(self, intp statistic_index, uint32 weight, bint remove):
        # Class members
        cdef LabelMatrix label_matrix = self.label_matrix
        cdef float64[::1, :] uncovered_labels = self.uncovered_labels
        cdef uint8[::1] minority_labels = self.minority_labels
        cdef float64[::1, :] confusion_matrices_subsample_default = self.confusion_matrices_subsample_default
        # The number of labels
        cdef intp num_labels = minority_labels.shape[0]
        # The given weight multiplied by 1 or -1, depending on the argument `remove`
        cdef float64 signed_weight = -<float64>weight if remove else weight
        # Temporary variables
        cdef uint8 uncovered, true_label, predicted_label
        cdef intp c

        for c in range(num_labels):
            uncovered = <uint8>uncovered_labels[statistic_index, c]

            # Only uncovered labels must be considered...
            if uncovered:
                true_label = label_matrix.get_label(statistic_index, c)
                predicted_label = minority_labels[c]

                # Add the current example and label to the confusion matrix for the current label...
                if true_label:
                    if predicted_label:
                        confusion_matrices_subsample_default[c, _RP] += signed_weight
                    else:
                        confusion_matrices_subsample_default[c, _RN] += signed_weight
                else:
                    if predicted_label:
                        confusion_matrices_subsample_default[c, _IP] += signed_weight
                    else:
                        confusion_matrices_subsample_default[c, _IN] += signed_weight

    cdef RefinementSearch begin_search(self, intp[::1] label_indices):
        # TODO
        pass

    cdef void apply_prediction(self, intp statistic_index, intp[::1] label_indices, HeadCandidate* head):
        # Class members
        cdef LabelMatrix label_matrix = self.label_matrix
        cdef float64[::1, :] uncovered_labels = self.uncovered_labels
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
