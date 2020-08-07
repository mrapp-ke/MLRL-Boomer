"""
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides classes that allow to store gradients and Hessians that are calculated according to a (non-decomposable) loss
function that is applied example-wise.
"""
from boomer.common._arrays cimport array_float64, c_matrix_float64, get_index
from boomer.boosting._math cimport triangular_number


cdef class ExampleWiseStatistics(GradientStatistics):
    """
    Allows to store gradients and Hessians that are calculated according to a loss function that is applied
    example-wise.
    """

    def __cinit__(self, ExampleWiseLoss loss_function, ExampleWiseRuleEvaluation rule_evaluation):
        """
        :param loss_function:   The loss function to be used for calculating gradients and Hessians
        :param rule_evaluation: The `ExampleWiseRuleEvaluation` to be used for calculating the predictions, as well as
                                corresponding quality scores, of rules
        """
        self.loss_function = loss_function
        self.rule_evaluation = rule_evaluation

    cdef void apply_default_prediction(self, LabelMatrix label_matrix, DefaultPrediction* default_prediction):
        # Class members
        cdef ExampleWiseLoss loss_function = self.loss_function
        # The number of examples
        cdef intp num_examples = label_matrix.num_examples
        # The number of labels
        cdef intp num_labels = label_matrix.num_labels
        # The number of hessians
        cdef intp num_hessians = triangular_number(num_labels)
        # A matrix that stores the currently predicted scores for each example and label
        cdef float64[:, ::1] current_scores = c_matrix_float64(num_examples, num_labels)
        # A matrix that stores the gradients for each example
        cdef float64[:, ::1] gradients = c_matrix_float64(num_examples, num_labels)
        # An array that stores the column-wise sums of the matrix of gradients
        cdef float64[::1] total_sums_of_gradients = array_float64(num_labels)
        # A matrix that stores the Hessians for each example
        cdef float64[:, ::1] hessians = c_matrix_float64(num_examples, num_hessians)
        # An array that stores the column-wise sums of the matrix of Hessians
        cdef float64[::1] total_sums_of_hessians = array_float64(num_hessians)
        # An array that stores the scores that are predicted by the default rule or NULL, if no default rule is used
        cdef float64* predicted_scores = default_prediction.predictedScores_ if default_prediction != NULL else NULL
        # Temporary variables
        cdef float64 predicted_score
        cdef intp r, c

        for r in range(num_examples):
            for c in range(num_labels):
                # Store the score that is predicted by the default rule for the current example and label...
                predicted_score = predicted_scores[c] if predicted_scores != NULL else 0
                current_scores[r, c] = predicted_scores[c]

            # Calculate the gradients and Hessians for the current example...
            loss_function.calculate_gradients_and_hessians(label_matrix, r, &current_scores[r, :][0], gradients[r, :],
                                                           hessians[r, :])

        # Store class members...
        self.gradients = gradients
        self.total_sums_of_gradients = total_sums_of_gradients
        self.hessians = hessians
        self.total_sums_of_hessians = total_sums_of_hessians
        self.label_matrix = label_matrix
        self.current_scores = current_scores

    cdef void reset_covered_statistics(self):
        # Class members
        cdef float64[::1] total_sums_of_gradients = self.total_sums_of_gradients
        cdef float64[::1] total_sums_of_hessians = self.total_sums_of_hessians
        # Reset total sums of gradients and Hessians to 0...
        total_sums_of_gradients[:] = 0
        total_sums_of_hessians[:] = 0

    cdef void update_covered_statistic(self, intp statistic_index, uint32 weight, bint remove):
        # Class members
        cdef float64[:, ::1] gradients = self.gradients
        cdef float64[::1] total_sums_of_gradients = self.total_sums_of_gradients
        cdef float64[:, ::1] hessians = self.hessians
        cdef float64[::1] total_sums_of_hessians = self.total_sums_of_hessians
        # The given weight multiplied by 1 or -1, depending on the argument `remove`
        cdef float64 signed_weight = -<float64>weight if remove else weight
        # The number of gradients/Hessians...
        cdef intp num_elements = gradients.shape[1]
        # Temporary variables
        cdef intp c

        # Add the gradients of the example at the given index (weighted by the given weight) to the total sums of
        # gradients...
        for c in range(num_elements):
            total_sums_of_gradients[c] += (signed_weight * gradients[statistic_index, c])

        # Add the Hessians of the example at the given index (weighted by the given weight) to the total sums of
        # Hessians...
        num_elements = hessians.shape[1]

        for c in range(num_elements):
            total_sums_of_hessians[c] += (signed_weight * hessians[statistic_index, c])

    cdef AbstractRefinementSearch* begin_search(self, intp[::1] label_indices):
        # Class members
        cdef ExampleWiseRuleEvaluation rule_evaluation = self.rule_evaluation
        cdef float64[:, ::1] gradients = self.gradients
        cdef float64[::1] total_sums_of_gradients = self.total_sums_of_gradients
        cdef float64[:, ::1] hessians = self.hessians
        cdef float64[::1] total_sums_of_hessians = self.total_sums_of_hessians

        # Instantiate and return a new object of the class `ExampleWiseRefinementSearchImpl`...
        cdef intp num_labels = total_sums_of_gradients.shape[0]
        cdef intp num_predictions = num_labels if label_indices is None else label_indices.shape[0]
        cdef const intp* label_indices_ptr = <const intp*>NULL if label_indices is None else &label_indices[0]
        return new ExampleWiseRefinementSearchImpl(rule_evaluation.rule_evaluation, num_predictions, label_indices_ptr,
                                                   num_labels, &gradients[0, 0], &total_sums_of_gradients[0],
                                                   &hessians[0, 0], &total_sums_of_hessians[0])

    cdef void apply_prediction(self, intp statistic_index, intp[::1] label_indices, HeadCandidate* head):
        # Class members
        cdef ExampleWiseLoss loss_function = self.loss_function
        cdef LabelMatrix label_matrix = self.label_matrix
        cdef float64[:, ::1] current_scores = self.current_scores
        cdef float64[:, ::1] gradients = self.gradients
        cdef float64[:, ::1] hessians = self.hessians
        # The number of predicted labels
        cdef intp num_predictions = head.numPredictions_
        # The predicted scores
        cdef float64* predicted_scores = head.predictedScores_
        # Temporary variables
        cdef intp c, l

        # Traverse the labels for which the new rule predicts to update the scores that are currently predicted for the
        # example at the given index...
        for c in range(num_predictions):
            l = get_index(c, label_indices)
            current_scores[statistic_index, l] += predicted_scores[c]

        # Update the gradients and Hessians for the example at the given index...
        loss_function.calculate_gradients_and_hessians(label_matrix, statistic_index,
                                                       &current_scores[statistic_index, :][0],
                                                       gradients[statistic_index, :], hessians[statistic_index, :])
