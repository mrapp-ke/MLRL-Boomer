"""
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides classes that allow to store gradients and Hessians that are calculated according to a loss function that is
applied label-wise.
"""
from boomer.common._arrays cimport array_float64, c_matrix_float64, get_index

from libc.stdlib cimport malloc

from libcpp.pair cimport pair


cdef class LabelWiseRefinementSearch(DecomposableRefinementSearch):
    """
    Allows to search for the best refinement of a rule based on the gradients and Hessians previously stored by
    `LabelWiseStatistics`.
    """

    def __cinit__(self, LabelWiseRuleEvaluation rule_evaluation, const intp[::1] label_indices,
                  const float64[:, ::1] gradients, const float64[::1] total_sums_of_gradients,
                  const float64[:, ::1] hessians, const float64[::1] total_sums_of_hessians):
        """
        :param rule_evaluation:         The `RuleEvaluation` to be used for calculating the predictions, as well as
                                        corresponding quality scores, of rules
        :param label_indices:           An array of dtype int, shape `(num_considered_labels)`, representing the indices
                                        of the labels that should be considered by the search or None, if all labels
                                        should be considered
        :param gradients:               An array of dtype float, shape `(num_examples, num_labels)`, representing the
                                        gradient for each example and label
        :param total_sums_of_gradients: An array of dtype float, shape `(num_labels)`, representing the sum of the
                                        gradients of all examples, which should be considered by the search, for each
                                        label
        :param hessians:                An array of dtype float, shape `(num_examples, num_labels)`, representing the
                                        Hessian for each example and label
        :param total_sums_of_hessians:  An array of dtype float, shape `(num_labels)`, representing the sum of the
                                        Hessians of all examples, which should be considered by the search, for each
                                        label
        """
        self.rule_evaluation = rule_evaluation
        self.label_indices = label_indices
        self.gradients = gradients
        self.total_sums_of_gradients = total_sums_of_gradients
        cdef intp num_labels = total_sums_of_gradients.shape[0] if label_indices is None else label_indices.shape[0]
        cdef float64[::1] sums_of_gradients = array_float64(num_labels)
        sums_of_gradients[:] = 0
        self.sums_of_gradients = sums_of_gradients
        self.accumulated_sums_of_gradients = None
        self.hessians = hessians
        self.total_sums_of_hessians = total_sums_of_hessians
        cdef float64[::1] sums_of_hessians = array_float64(num_labels)
        sums_of_hessians[:] = 0
        self.sums_of_hessians = sums_of_hessians
        self.accumulated_sums_of_hessians = None
        cdef float64* predicted_scores = <float64*>malloc(num_labels * sizeof(float64))
        cdef float64* quality_scores = <float64*>malloc(num_labels * sizeof(float64))
        cdef LabelWisePrediction* prediction = new LabelWisePrediction(num_labels, predicted_scores, quality_scores, 0)
        self.prediction = prediction

    def __dealloc__(self):
        del self.prediction

    cdef void update_search(self, intp statistic_index, uint32 weight):
        # Class members
        cdef const float64[:, ::1] gradients = self.gradients
        cdef float64[::1] sums_of_gradients = self.sums_of_gradients
        cdef const float64[:, ::1] hessians = self.hessians
        cdef float64[::1] sums_of_hessians = self.sums_of_hessians
        cdef const intp[::1] label_indices = self.label_indices
        # The number of labels considered by the current search
        cdef intp num_labels = sums_of_gradients.shape[0]
        # Temporary variables
        cdef intp c, l

        # For each label, add the gradient and Hessian of the example at the given index (weighted by the given weight)
        # to the current sum of gradients and Hessians...
        for c in range(num_labels):
            l = get_index(c, label_indices)
            sums_of_gradients[c] += (weight * gradients[statistic_index, l])
            sums_of_hessians[c] += (weight * hessians[statistic_index, l])

    cdef void reset_search(self):
        # Class members
        cdef float64[::1] sums_of_gradients = self.sums_of_gradients
        cdef float64[::1] sums_of_hessians = self.sums_of_hessians
        # The number of labels
        cdef intp num_labels = sums_of_gradients.shape[0]
        # Temporary variables
        cdef intp c

        # Update the arrays that store the accumulated sums of gradients and hessians...
        cdef float64[::1] accumulated_sums_of_gradients = self.accumulated_sums_of_gradients
        cdef float64[::1] accumulated_sums_of_hessians

        if accumulated_sums_of_gradients is None:
            accumulated_sums_of_gradients = array_float64(num_labels)
            self.accumulated_sums_of_gradients = accumulated_sums_of_gradients
            accumulated_sums_of_hessians = array_float64(num_labels)
            self.accumulated_sums_of_hessians = accumulated_sums_of_hessians

            for c in range(num_labels):
                accumulated_sums_of_gradients[c] = sums_of_gradients[c]
                sums_of_gradients[c] = 0
                accumulated_sums_of_hessians[c] = sums_of_hessians[c]
                sums_of_hessians[c] = 0
        else:
            accumulated_sums_of_hessians = self.accumulated_sums_of_hessians

            for c in range(num_labels):
                accumulated_sums_of_gradients[c] += sums_of_gradients[c]
                sums_of_gradients[c] = 0
                accumulated_sums_of_hessians[c] += sums_of_hessians[c]
                sums_of_hessians[c] = 0

    cdef LabelWisePrediction* calculate_label_wise_prediction(self, bint uncovered, bint accumulated):
        # Class members
        cdef LabelWiseRuleEvaluation rule_evaluation = self.rule_evaluation
        cdef LabelWisePrediction* prediction = self.prediction
        cdef const intp[::1] label_indices = self.label_indices
        cdef const float64[::1] total_sums_of_gradients = self.total_sums_of_gradients
        cdef float64[::1] sums_of_gradients = self.accumulated_sums_of_gradients if accumulated else self.sums_of_gradients
        cdef const float64[::1] total_sums_of_hessians = self.total_sums_of_hessians
        cdef float64[::1] sums_of_hessians = self.accumulated_sums_of_hessians if accumulated else self.sums_of_hessians

        # Calculate and return the predictions, as well as corresponding quality scores...
        rule_evaluation.calculate_label_wise_prediction(label_indices, total_sums_of_gradients, sums_of_gradients,
                                                        total_sums_of_hessians, sums_of_hessians, uncovered, prediction)
        return prediction


cdef class LabelWiseStatistics(GradientStatistics):
    """
    Allows to store gradients and hessians that are calculated according to a loss function that is applied label-wise.
    """

    def __cinit__(self, LabelWiseLossFunction loss_function, LabelWiseRuleEvaluation rule_evaluation):
        """
        :param loss_function:   The loss function to be minimized
        :param rule_evaluation: The `RuleEvaluation` to be used for calculating the predictions, as well as
                                corresponding quality scores, of rules
        """
        self.loss_function = loss_function
        self.rule_evaluation = rule_evaluation

    cdef void apply_default_prediction(self, LabelMatrix label_matrix, DefaultPrediction* default_prediction):
        # Class members
        cdef LabelWiseLossFunction loss_function = self.loss_function
        # The number of examples
        cdef intp num_examples = label_matrix.num_examples
        # The number of labels
        cdef intp num_labels = label_matrix.num_labels
        # An array that stores the predictions of the default rule
        cdef float64* predicted_scores = default_prediction.predictedScores_ if default_prediction != NULL else NULL
        # A matrix that stores the currently predicted scores for each example and label
        cdef float64[:, ::1] current_scores = c_matrix_float64(num_examples, num_labels)
        # A matrix that stores the gradients for each example and label
        cdef float64[:, ::1] gradients = c_matrix_float64(num_examples, num_labels)
        # An array that stores the column-wise sums of the matrix of gradients
        cdef float64[::1] total_sums_of_gradients = array_float64(num_labels)
        # A matrix that stores the Hessians for each example and label
        cdef float64[:, ::1] hessians = c_matrix_float64(num_examples, num_labels)
        # An array that stores the column-wise sums of the matrix of hessians
        cdef float64[::1] total_sums_of_hessians = array_float64(num_labels)
        # Temporary variables
        cdef pair[float64, float64] gradient_and_hessian
        cdef float64 predicted_score, gradient, hessian
        cdef intp c, r

        for c in range(num_labels):
            predicted_score = predicted_scores[c] if predicted_scores != NULL else 0

            for r in range(num_examples):
                # Calculate the gradient and Hessian for the current example and label...
                gradient_and_hessian = loss_function.calculate_gradient_and_hessian(label_matrix, r, c, predicted_score)
                gradient = gradient_and_hessian.first
                gradients[r, c] = gradient
                hessian = gradient_and_hessian.second
                hessians[r, c] = hessian

                # Store the score that is currently predicted for the current example and label...
                current_scores[r, c] = predicted_score

        # Store the gradients...
        self.gradients = gradients
        self.total_sums_of_gradients = total_sums_of_gradients

        # Store the Hessians...
        self.hessians = hessians
        self.total_sums_of_hessians = total_sums_of_hessians

        # Store the label matrix and the currently predicted scores...
        self.label_matrix = label_matrix
        self.current_scores = current_scores

    cdef void reset_statistics(self):
        # Class members
        cdef float64[::1] total_sums_of_gradients = self.total_sums_of_gradients
        cdef float64[::1] total_sums_of_hessians = self.total_sums_of_hessians
        # The number of labels
        cdef intp num_labels = total_sums_of_gradients.shape[0]
        # Temporary variables
        cdef intp c

        # Reset total sums of gradients and Hessians to 0...
        for c in range(num_labels):
            total_sums_of_gradients[c] = 0
            total_sums_of_hessians[c] = 0

    cdef void update_covered_statistic(self, intp statistic_index, uint32 weight, bint remove):
        # Class members
        cdef float64[:, ::1] gradients = self.gradients
        cdef float64[::1] total_sums_of_gradients = self.total_sums_of_gradients
        cdef float64[:, ::1] hessians = self.hessians
        cdef float64[::1] total_sums_of_hessians = self.total_sums_of_hessians
        # The number of labels
        cdef intp num_labels = total_sums_of_gradients.shape[0]
        # The given weight multiplied by 1 or -1, depending on the argument `remove`
        cdef float64 signed_weight = -<float64>weight if remove else weight
        # Temporary variables
        cdef intp c

        # For each label, add the gradient and Hessian of the example at the given index (weighted by the given weight)
        # to the total sums of gradients and Hessians...
        for c in range(num_labels):
            total_sums_of_gradients[c] += (signed_weight * gradients[statistic_index, c])
            total_sums_of_hessians[c] += (signed_weight * hessians[statistic_index, c])

    cdef RefinementSearch begin_search(self, intp[::1] label_indices):
        # Class members
        cdef LabelWiseRuleEvaluation rule_evaluation = self.rule_evaluation
        cdef float64[:, ::1] gradients = self.gradients
        cdef float64[::1] total_sums_of_gradients = self.total_sums_of_gradients
        cdef float64[:, ::1] hessians = self.hessians
        cdef float64[::1] total_sums_of_hessians = self.total_sums_of_hessians

        # Instantiate and return a new object of the class `LabelWiseRefinementSearch`...
        return LabelWiseRefinementSearch.__new__(LabelWiseRefinementSearch, rule_evaluation, label_indices, gradients,
                                                 total_sums_of_gradients, hessians, total_sums_of_hessians)

    cdef void apply_prediction(self, intp statistic_index, intp[::1] label_indices, HeadCandidate* head):
        # Class members
        cdef LabelWiseLossFunction loss_function = self.loss_function
        cdef LabelMatrix label_matrix = self.label_matrix
        cdef float64[:, ::1] current_scores = self.current_scores
        cdef float64[:, ::1] gradients = self.gradients
        cdef float64[:, ::1] hessians = self.hessians
        # The number of predicted labels
        cdef intp num_predictions = head.numPredictions_
        # The predicted scores
        cdef float64* predicted_scores = head.predictedScores_
        # Temporary variables
        cdef pair[float64, float64] gradient_and_hessian
        cdef float64 predicted_score, updated_score, gradient, hessian
        cdef intp c, l

        # Only the labels that are predicted by the new rule must be considered...
        for c in range(num_predictions):
            l = get_index(c, label_indices)
            predicted_score = predicted_scores[c]

            # Update the score that is currently predicted for the current example and label...
            updated_score = current_scores[statistic_index, l] + predicted_score
            current_scores[statistic_index, l] = updated_score

            # Update the gradient and Hessian for the current example and label...
            gradient_and_hessian = loss_function.calculate_gradient_and_hessian(label_matrix, statistic_index, l,
                                                                                updated_score)
            gradient = gradient_and_hessian.first
            gradients[statistic_index, l] = gradient
            hessian = gradient_and_hessian.second
            hessians[statistic_index, l] = hessian
