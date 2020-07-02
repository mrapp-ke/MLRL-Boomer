"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides classes that implement loss functions that are applied example- and label-wise.
"""
from boomer.common._arrays cimport array_float64, c_matrix_float64, get_index
from boomer.boosting.differentiable_losses cimport _convert_label_into_score, _l2_norm_pow

from libc.math cimport pow, exp


cdef class LabelWiseLossFunction:
    """
    A base class for all differentiable loss functions that are applied label-wise.
    """

    cdef float64 gradient(self, uint8 true_label, float64 predicted_score):
        """
        Must be implemented by subclasses to calculate the gradient (first derivative) of the loss function for a
        certain example and label.

        :param true_label:      A scalar of dtype uint8, representing the true label according to the ground truth
        :param predicted_score: A scalar of dtype float64, representing the score that is predicted for the respective
                                example and label
        :return:                A scalar of dtype float64, representing the gradient that has been calculated
        """
        pass

    cdef float64 hessian(self, uint8 true_label, float64 predicted_score):
        """
        Must be implemented by subclasses to calculate the hessian (second derivative) of the loss function for a
        certain example and label.

        :param true_label:      A scalar of dtype uint8, representing the true label according to the ground truth
        :param predicted_score: A scalar of dtype float64, representing the score that is predicted for the respective
                                example and label
        :return:                A scalar of dtype float64, representing the hessian that has been calculated
        """
        pass


cdef class LabelWiseLogisticLossFunction(LabelWiseLossFunction):
    """
    A multi-label variant of the logistic loss that is applied label-wise.
    """

    cdef float64 gradient(self, uint8 true_label, float64 predicted_score):
        cdef float64 expected_score = 1 if true_label else -1
        return -expected_score / (1 + exp(expected_score * predicted_score))

    cdef float64 hessian(self, uint8 true_label, float64 predicted_score):
        cdef float64 expected_score = 1 if true_label else -1
        cdef float64 exponential = exp(expected_score * predicted_score)
        return (pow(expected_score, 2) * exponential) / pow(1 + exponential, 2)


cdef class LabelWiseSquaredErrorLossFunction(LabelWiseLossFunction):
    """
    A multi-label variant of the squared error loss that is applied label-wise.
    """

    cdef float64 gradient(self, uint8 true_label, float64 predicted_score):
        cdef float64 expected_score = 1 if true_label else -1
        return 2 * predicted_score - 2 * expected_score

    cdef float64 hessian(self, uint8 true_label, float64 predicted_score):
        return 2


cdef class LabelWiseRefinementSearch(DecomposableRefinementSearch):
    """
    Allows to search for the best refinement of a rule according to a differentiable loss function that is applied
    label-wise.
    """

    def __cinit__(self, float64 l2_regularization_weight, const intp[::1] label_indices,
                  const float64[:, ::1] gradients, const float64[::1] total_sums_of_gradients,
                  const float64[:, ::1] hessians, const float64[::1] total_sums_of_hessians):
        """
        :param l2_regularization_weight:    The weight of the L2 regularization that is applied for calculating the
                                            optimal scores to be predicted by rules
        :param label_indices:               An array of dtype int, shape `(num_considered_labels)`, representing the
                                            indices of the labels that should be considered by the search or None, if
                                            all labels should be considered
        :param gradients:                   An array of dtype float, shape `(num_examples, num_labels)`, representing
                                            the gradient for each example and label
        :param total_sums_of_gradients:     An array of dtype float, shape `(num_labels)`, representing the sum of the
                                            gradients of all examples, which should be considered by the search, for
                                            each label
        :param hessians:                    An array of dtype float, shape `(num_examples, num_labels)`, representing
                                            the hessian for each example and label
        :param total_sums_of_hessians:      An array of dtype float, shape `(num_labels)`, representing the sum of the
                                            hessians of all examples, which should be considered by the search, for each
                                            label
        """
        self.l2_regularization_weight = l2_regularization_weight
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
        cdef LabelWisePrediction prediction = LabelWisePrediction.__new__(LabelWisePrediction)
        cdef float64[::1] predicted_scores = array_float64(num_labels)
        prediction.predicted_scores = predicted_scores
        cdef float64[::1] quality_scores = array_float64(num_labels)
        prediction.quality_scores = quality_scores
        self.prediction = prediction

    cdef void update_search(self, intp example_index, uint32 weight):
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

        # For each label, add the gradient and hessian of the example at the given index (weighted by the given weight)
        # to the current sum of gradients and hessians...
        for c in range(num_labels):
            l = get_index(c, label_indices)
            sums_of_gradients[c] += (weight * gradients[example_index, l])
            sums_of_hessians[c] += (weight * hessians[example_index, l])

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

    cdef LabelWisePrediction calculate_label_wise_prediction(self, bint uncovered, bint accumulated):
        # Class members
        cdef float64 l2_regularization_weight = self.l2_regularization_weight
        cdef LabelWisePrediction prediction = self.prediction
        cdef float64[::1] predicted_scores = prediction.predicted_scores
        cdef float64[::1] quality_scores = prediction.quality_scores
        cdef float64[::1] sums_of_gradients = self.accumulated_sums_of_gradients if accumulated else self.sums_of_gradients
        cdef float64[::1] sums_of_hessians = self.accumulated_sums_of_hessians if accumulated else self.sums_of_hessians
        # The number of labels considered by the current search
        cdef intp num_labels = sums_of_gradients.shape[0]
        # The overall quality score, i.e., the sum of the quality scores for each label plus the L2 regularization term
        cdef float64 overall_quality_score = 0
        # Temporary variables
        cdef const float64[::1] total_sums_of_gradients, total_sums_of_hessians
        cdef const intp[::1] label_indices
        cdef float64 sum_of_gradients, sum_of_hessians, score, score_pow
        cdef intp c, l

        if uncovered:
            total_sums_of_gradients = self.total_sums_of_gradients
            total_sums_of_hessians = self.total_sums_of_hessians
            label_indices = self.label_indices

        # For each label, calculate the score to be predicted, as well as a quality score...
        for c in range(num_labels):
            sum_of_gradients = sums_of_gradients[c]
            sum_of_hessians = sums_of_hessians[c]

            if uncovered:
                l = get_index(c, label_indices)
                sum_of_gradients = total_sums_of_gradients[l] - sum_of_gradients
                sum_of_hessians = total_sums_of_hessians[l] - sum_of_hessians

            # Calculate score to be predicted for the current label...
            score = sum_of_hessians + l2_regularization_weight
            score = -sum_of_gradients / score if score != 0 else 0
            predicted_scores[c] = score

            # Calculate the quality score for the current label...
            score_pow = pow(score, 2)
            score = (sum_of_gradients * score) + (0.5 * score_pow * sum_of_hessians)
            quality_scores[c] = score + (0.5 * l2_regularization_weight * score_pow)
            overall_quality_score += score

        # Add the L2 regularization term to the overall quality score...
        overall_quality_score += 0.5 * l2_regularization_weight * _l2_norm_pow(predicted_scores)
        prediction.overall_quality_score = overall_quality_score

        return prediction


cdef class LabelWiseDifferentiableLoss(DifferentiableLoss):
    """
    Allows to locally minimize a differentiable (surrogate) loss function that is applied label-wise by the rules that
    are learned by a boosting algorithm.
    """

    def __cinit__(self, LabelWiseLossFunction loss_function, float64 l2_regularization_weight):
        """
        :param loss_function:               A label-wise differentiable loss function to be minimized
        :param l2_regularization_weight:    The weight of the L2 regularization that is applied for calculating the
                                            optimal scores to be predicted by rules. Increasing this value causes the
                                            model to be more conservative, setting it to 0 turns of L2 regularization
                                            entirely
        """
        self.loss_function = loss_function
        self.l2_regularization_weight = l2_regularization_weight

    cdef DefaultPrediction calculate_default_prediction(self, uint8[:, ::1] y):
        # A label-wise differentiable loss function to be minimized
        cdef LabelWiseLossFunction loss_function = self.loss_function
        # The weight to be used for L2 regularization
        cdef float64 l2_regularization_weight = self.l2_regularization_weight
        # The number of examples
        cdef intp num_examples = y.shape[0]
        # The number of labels
        cdef intp num_labels = y.shape[1]
        # A matrix that stores the currently predicted scores for each example and label
        cdef float64[:, ::1] current_scores = c_matrix_float64(num_examples, num_labels)
        # A matrix that stores the gradients for each example and label
        cdef float64[:, ::1] gradients = c_matrix_float64(num_examples, num_labels)
        # An array that stores the column-wise sums of the matrix of gradients
        cdef float64[::1] total_sums_of_gradients = array_float64(num_labels)
        # A matrix that stores the hessians for each example and label
        cdef float64[:, ::1] hessians = c_matrix_float64(num_examples, num_labels)
        # An array that stores the column-wise sums of the matrix of hessians
        cdef float64[::1] total_sums_of_hessians = array_float64(num_labels)
        # An array that stores the scores that are predicted by the default rule
        cdef float64[::1] predicted_scores = array_float64(num_labels)
        cdef DefaultPrediction prediction = DefaultPrediction.__new__(DefaultPrediction)
        prediction.predicted_scores = predicted_scores
        # Temporary variables
        cdef float64 sum_of_gradients, sum_of_hessians, predicted_score, tmp
        cdef uint8 true_label
        cdef intp c, r

        for c in range(num_labels):
            # Column-wise sum up the gradients and hessians for the current label...
            sum_of_gradients = 0
            sum_of_hessians = 0

            for r in range(num_examples):
                true_label = y[r, c]

                # Calculate gradient for the current example and label...
                tmp = loss_function.gradient(true_label, 0)
                sum_of_gradients += tmp

                # Calculate hessian for the current example and label...
                tmp = loss_function.hessian(true_label, 0)
                sum_of_hessians += tmp

            # Calculate optimal score to be predicted by the default rule for the current label...
            predicted_score = -sum_of_gradients / (sum_of_hessians + l2_regularization_weight)
            predicted_scores[c] = predicted_score

            # Traverse column again to calculate updated gradients based on the calculated score...
            for r in range(num_examples):
                true_label = y[r, c]

                # Calculate updated gradient for the current example and label...
                tmp = loss_function.gradient(true_label, predicted_score)
                gradients[r, c] = tmp

                # Calculate updated gradient for the current example and label...
                tmp = loss_function.hessian(true_label, predicted_score)
                hessians[r, c] = tmp

                # Store the score that is currently predicted for the current example and label...
                current_scores[r, c] = predicted_score

        # Store the gradients...
        self.gradients = gradients
        self.total_sums_of_gradients = total_sums_of_gradients

        # Store the hessians...
        self.hessians = hessians
        self.total_sums_of_hessians = total_sums_of_hessians

        # Store the true labels and the currently predicted scores...
        self.true_labels = y
        self.current_scores = current_scores

        return prediction

    cdef void begin_instance_sub_sampling(self):
        # Class members
        cdef float64[::1] total_sums_of_gradients = self.total_sums_of_gradients
        cdef float64[::1] total_sums_of_hessians = self.total_sums_of_hessians
        # The number of labels
        cdef intp num_labels = total_sums_of_gradients.shape[0]
        # Temporary variables
        cdef intp c

        # Reset total sums of gradients and hessians to 0...
        for c in range(num_labels):
            total_sums_of_gradients[c] = 0
            total_sums_of_hessians[c] = 0

    cdef void update_sub_sample(self, intp example_index, uint32 weight, bint remove):
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

        # For each label, add the gradient and hessian of the example at the given index (weighted by the given weight)
        # to the total sums of gradients and hessians...
        for c in range(num_labels):
            total_sums_of_gradients[c] += (signed_weight * gradients[example_index, c])
            total_sums_of_hessians[c] += (signed_weight * hessians[example_index, c])

    cdef RefinementSearch begin_search(self, intp[::1] label_indices):
        cdef float64 l2_regularization_weight = self.l2_regularization_weight
        cdef float64[:, ::1] gradients = self.gradients
        cdef float64[::1] total_sums_of_gradients = self.total_sums_of_gradients
        cdef float64[:, ::1] hessians = self.hessians
        cdef float64[::1] total_sums_of_hessians = self.total_sums_of_hessians
        return LabelWiseRefinementSearch.__new__(LabelWiseRefinementSearch, l2_regularization_weight, label_indices,
                                                 gradients, total_sums_of_gradients, hessians, total_sums_of_hessians)

    cdef void apply_prediction(self, intp example_index, intp[::1] label_indices, float64[::1] predicted_scores):
        # Class members
        cdef LabelWiseLossFunction loss_function = self.loss_function
        cdef uint8[:, ::1] true_labels = self.true_labels
        cdef float64[:, ::1] current_scores = self.current_scores
        cdef float64[:, ::1] gradients = self.gradients
        cdef float64[:, ::1] hessians = self.hessians
        # The number of predicted labels
        cdef intp num_labels = predicted_scores.shape[0]
        # Temporary variables
        cdef float64 predicted_score, current_score
        cdef uint8 true_label
        cdef intp c, l

        # Only the labels that are predicted by the new rule must be considered...
        for c in range(num_labels):
            l = get_index(c, label_indices)
            true_label = true_labels[example_index, l]
            predicted_score = predicted_scores[c]

            # Update the score that is currently predicted for the current example and label...
            current_score = current_scores[example_index, l] + predicted_score
            current_scores[example_index, l] = current_score

            # Update the gradient for the current example and label...
            gradients[example_index, l] = loss_function.gradient(true_label, current_score)

            # Update the hessian for the current example and label...
            hessians[example_index, l] = loss_function.hessian(true_label, current_score)
