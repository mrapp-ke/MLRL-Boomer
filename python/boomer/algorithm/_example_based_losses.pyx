# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides classes that implement loss functions that are applied example-wise.
"""
from boomer.algorithm._arrays cimport array_float64, matrix_float64
from boomer.algorithm._utils cimport get_index, convert_label_into_score
from boomer.algorithm._math cimport divide_or_zero_float64, triangular_number, l2_norm_pow
from boomer.algorithm._math cimport dsysv_float64, dspmv_float64, ddot_float64

from libc.math cimport pow, exp, fabs


cdef class ExampleBasedLogisticLoss(NonDecomposableLoss):
    """
    A multi-label variant of the logistic loss that is applied example-wise.
    """

    def __cinit__(self, float64 l2_regularization_weight):
        """
        :param l2_regularization_weight: The weight of the L2 regularization that is applied for calculating the optimal
                                         scores to be predicted by rules. Increasing this value causes the model to be
                                         more conservative, setting it to 0 turns of L2 regularization entirely
        """
        self.l2_regularization_weight = l2_regularization_weight
        self.prediction = LabelIndependentPrediction()

    cdef float64[::1] calculate_default_scores(self, uint8[::1, :] y):
        cdef float64 l2_regularization_weight = self.l2_regularization_weight
        cdef intp num_rows = y.shape[0]
        cdef intp num_cols = y.shape[1]
        cdef float64 sum_of_exponentials = num_cols + 1
        cdef float64 sum_of_exponentials_pow = pow(sum_of_exponentials, 2)
        cdef float64[::1] expected_scores = array_float64(num_cols)
        cdef float64 expected_score
        cdef intp r, c, c2, i

        # We find the optimal scores to be predicted by the default rule for each label by solving a system of linear
        # equations A * X = B with one equation per label. A is a two-dimensional (symmetrical) matrix of coefficients,
        # B is an one-dimensional array of ordinates, and X is an one-dimensional array of unknowns to be determined.
        # The ordinates result from the gradients of the loss function, whereas the coefficients result from the
        # hessians. As the matrix of coefficients is symmetrical, we must only compute the hessians that correspond to
        # the upper-right triangle of the matrix and leave the remaining elements unspecified. For reasons of space
        # efficiency, we store the hessians in an one-dimensional array by appending the columns of the matrix of
        # coefficients to each other and omitting the unspecified elements.
        cdef float64[::1] ordinates = array_float64(num_cols)
        ordinates[:] = 0
        cdef intp num_hessians = triangular_number(num_cols)  # The number of elements in the upper-right triangle
        cdef float64[::1] coefficients = array_float64(num_hessians)
        coefficients[:] = 0

        # Example-wise calculate the gradients and hessians and add them to the arrays of ordinates and coefficients...
        for r in range(num_rows):
            # Traverse the labels of the current example once to create an array of expected scores that is shared among
            # the upcoming calculations of gradients and hessians...
            for c in range(num_cols):
                expected_scores[c] = convert_label_into_score(y[r, c])

            # Traverse the labels again to calculate the gradients and hessians...
            i = 0

            for c in range(num_cols):
                expected_score = expected_scores[c]

                # Calculate the first derivative (gradient) of the loss function with respect to the current label and
                # add it to the array of ordinates...
                ordinates[c] += expected_score / sum_of_exponentials

                # Calculate the second derivatives (hessians) of the loss function with respect to the current label and
                # each of the other labels and add it to the matrix of coefficients...
                for c2 in range(c):
                    coefficients[i] -= (expected_scores[c2] * expected_score) / sum_of_exponentials_pow
                    i += 1

                # Calculate the second derivative (hessian) of the loss function with respect to the current label and
                # add it to the diagonal of the matrix of coefficients...
                coefficients[i] += (fabs(expected_score) * num_cols) / sum_of_exponentials_pow
                i += 1

        # Compute the optimal scores to be predicted by the default rule by solving the system of linear equations...
        cdef float64[::1] scores = dsysv_float64(coefficients, ordinates, l2_regularization_weight)

        # We must traverse each example again to calculate the updated gradients and hessians based on the calculated
        # scores...
        cdef float64[::1] exponentials = ordinates # Reuse existing array instead of allocating a new one
        cdef float64[::1, :] gradients = matrix_float64(num_rows, num_cols)
        cdef float64[::1] total_sums_of_gradients = array_float64(num_cols)
        total_sums_of_gradients[:] = 0
        cdef float64[::1, :] hessians = matrix_float64(num_rows, num_hessians)
        cdef float64[::1] total_sums_of_hessians = coefficients # Reuse existing array instead of allocating a new one
        total_sums_of_hessians[:] = 0
        cdef float64[::1, :] current_scores = matrix_float64(num_rows, num_cols)
        cdef float64 exponential, tmp, score

        for r in range(num_rows):
            # Traverse the labels of the current example once to create arrays of expected scores and exponentials that
            # are shared among the upcoming calculations of gradients and hessians...
            sum_of_exponentials = 1

            for c in range(num_cols):
                expected_score = convert_label_into_score(y[r, c])
                expected_scores[c] = expected_score
                exponential = exp(-expected_score * scores[c])
                exponentials[c] = exponential
                sum_of_exponentials += exponential

            sum_of_exponentials_pow = pow(sum_of_exponentials, 2)

            # Traverse the labels again to calculate the gradients and hessians...
            i = 0

            for c in range(num_cols):
                expected_score = expected_scores[c]
                exponential = exponentials[c]
                score = scores[c]
                current_scores[r, c] = score

                # Calculate the first derivative (gradient) of the loss function with respect to the current label and
                # add it to the matrix of gradients...
                tmp = (expected_score * exponential) / sum_of_exponentials
                # Note: The sign of the gradient is inverted (from negative to positive), because otherwise, when using
                # the sums of gradients as the ordinates for solving a system of linear equations in the function
                # `evaluate_label_dependent_predictions`, the sign must be inverted again...
                gradients[r, c] = tmp
                total_sums_of_gradients[c] += tmp

                # Calculate the second derivatives (hessians) of the loss function with respect to the current label and
                # each of the other labels and add them to the matrix of hessians...
                for c2 in range(c):
                    tmp = exp(-expected_scores[c2] * scores[c2] - expected_score * score)
                    tmp = (expected_scores[c2] * expected_score * tmp) / sum_of_exponentials_pow
                    hessians[r, i] = -tmp
                    total_sums_of_hessians[i] -= tmp
                    i += 1

                # Calculate the second derivative (hessian) of the loss function with respect to the current label and
                # add it to the diagonal of the matrix of hessians...
                tmp = (fabs(expected_score) * exponential * (sum_of_exponentials - exponential)) / sum_of_exponentials_pow
                hessians[r, i] = tmp
                total_sums_of_hessians[i] += tmp
                i += 1

        # Cache the ground truth label matrix...
        self.ground_truth = y

        # Cache the matrix of currently predicted scores...
        self.current_scores = current_scores

        # Cache the matrix of gradients...
        self.gradients = gradients

        # Cache the total sums of gradients...
        self.total_sums_of_gradients = total_sums_of_gradients

        # Cache the matrix of hessians...
        self.hessians = hessians

        # Caches the total sums of hessians...
        self.total_sums_of_hessians = total_sums_of_hessians

        return scores

    cdef begin_instance_sub_sampling(self):
        # Reset the total sums of gradients and hessians to 0...
        cdef float64[::1] total_sums_of_gradients = self.total_sums_of_gradients
        total_sums_of_gradients[:] = 0
        cdef float64[::1] total_sums_of_hessians = self.total_sums_of_hessians
        total_sums_of_hessians[:] = 0

    cdef update_sub_sample(self, intp example_index):
        # Update the total sums of gradients...
        cdef float64[::1, :] gradients = self.gradients
        cdef float64[::1] total_sums_of_gradients = self.total_sums_of_gradients
        cdef intp num_cols = gradients.shape[1]
        cdef intp c

        for c in range(num_cols):
            total_sums_of_gradients[c] += gradients[example_index, c]

        # Update the total sums of hessians...
        cdef float64[::1, :] hessians = self.hessians
        cdef float64[::1] total_sums_of_hessians = self.total_sums_of_hessians
        num_cols = hessians.shape[1]

        for c in range(num_cols):
            total_sums_of_hessians[c] += hessians[example_index, c]

    cdef begin_search(self, intp[::1] label_indices):
        # Reset sums of gradients and hessians to 0...
        cdef float64[::1, :] gradients
        cdef intp num_gradients
        cdef float64[::1, :] hessians
        cdef intp num_hessians

        if label_indices is None:
            gradients = self.gradients
            num_gradients = gradients.shape[1]
            hessians = self.hessians
            num_hessians = hessians.shape[1]
        else:
            num_gradients = label_indices.shape[0]
            num_hessians = triangular_number(num_gradients)

        cdef float64[::1] sums_of_gradients = array_float64(num_gradients)
        sums_of_gradients[:] = 0
        cdef float64[::1] sums_of_hessians = array_float64(num_hessians)
        sums_of_hessians[:] = 0
        self.sums_of_gradients = sums_of_gradients
        self.sums_of_hessians = sums_of_hessians
        self.label_indices = label_indices

    cdef update_search(self, intp example_index, uint32 weight):
        cdef float64[::1, :] gradients = self.gradients
        cdef float64[::1, :] hessians = self.hessians
        cdef float64[::1] sums_of_gradients = self.sums_of_gradients
        cdef float64[::1] sums_of_hessians = self.sums_of_hessians
        cdef intp num_gradients = sums_of_gradients.shape[0]
        cdef intp[::1] label_indices = self.label_indices
        cdef intp i = 0
        cdef intp c, c2, l, l2, offset

        for c in range(num_gradients):
            l = get_index(c, label_indices)
            sums_of_gradients[c] += (weight * gradients[example_index, l])
            offset = triangular_number(l)

            for c2 in range(c + 1):
                l2 = offset + get_index(c2, label_indices)
                sums_of_hessians[i] += (weight * hessians[example_index, l2])
                i += 1

    cdef LabelIndependentPrediction evaluate_label_independent_predictions(self, bint uncovered):
        cdef float64 l2_regularization_weight = self.l2_regularization_weight
        cdef float64[::1] sums_of_gradients = self.sums_of_gradients
        cdef intp num_gradients = sums_of_gradients.shape[0]
        cdef float64[::1] sums_of_hessians = self.sums_of_hessians
        cdef LabelIndependentPrediction prediction = self.prediction
        cdef float64[::1] predicted_scores = prediction.predicted_scores
        cdef float64[::1] quality_scores = prediction.quality_scores

        if predicted_scores is None or num_gradients != predicted_scores.shape[0]:
            predicted_scores = array_float64(num_gradients)
            prediction.predicted_scores = predicted_scores

        if quality_scores is None or num_gradients != quality_scores.shape[0]:
            quality_scores = array_float64(num_gradients)
            prediction.quality_scores = quality_scores

        cdef float64 overall_quality_score = 0
        cdef float64[::1] total_sums_of_gradients, total_sums_of_hessians
        cdef intp[::1] label_indices
        cdef float64 sum_of_gradients, sum_of_hessians, score, score_pow
        cdef intp c, c2, l, l2

        if uncovered:
            total_sums_of_gradients = self.total_sums_of_gradients
            total_sums_of_hessians = self.total_sums_of_hessians
            label_indices = self.label_indices

        for c in range(num_gradients):
            sum_of_gradients = sums_of_gradients[c]
            c2 = triangular_number(c + 1) - 1
            sum_of_hessians = sums_of_hessians[c2]

            if uncovered:
                l = get_index(c, label_indices)
                sum_of_gradients = total_sums_of_gradients[l] - sum_of_gradients
                l2 = triangular_number(l + 1) - 1
                sum_of_hessians = total_sums_of_hessians[l2] - sum_of_hessians

            # Calculate predicted score...
            # Note: As the sign of the gradients was inverted in the function `calculate_default_scores`, it must be
            # reverted again in the following.
            score = divide_or_zero_float64(sum_of_gradients, sum_of_hessians + l2_regularization_weight)
            predicted_scores[c] = score

            # Calculate quality score...
            score_pow = pow(score, 2)
            score = (-sum_of_gradients * score) + (0.5 * score_pow * sum_of_hessians)
            quality_scores[c] = score + (0.5 * l2_regularization_weight * score_pow)
            overall_quality_score += score

        overall_quality_score += 0.5 * l2_regularization_weight * l2_norm_pow(predicted_scores)
        prediction.overall_quality_score = overall_quality_score
        return prediction

    cdef Prediction evaluate_label_dependent_predictions(self, bint uncovered):
        cdef float64 l2_regularization_weight = self.l2_regularization_weight
        cdef float64[::1] sums_of_gradients = self.sums_of_gradients
        cdef float64[::1] sums_of_hessians = self.sums_of_hessians
        cdef intp num_gradients = sums_of_gradients.shape[0]
        cdef Prediction prediction = self.prediction
        cdef float64[::1] gradients, hessians, total_sums_of_gradients, total_sums_of_hessians
        cdef intp[::1] label_indices
        cdef intp num_hessians, c, c2, l, l2, i, offset

        if uncovered:
            label_indices = self.label_indices
            num_hessians = sums_of_hessians.shape[0]
            gradients = array_float64(num_gradients)
            hessians = array_float64(num_hessians)
            total_sums_of_gradients = self.total_sums_of_gradients
            total_sums_of_hessians = self.total_sums_of_hessians
            i = 0

            for c in range(num_gradients):
                l = get_index(c, label_indices)
                gradients[c] = total_sums_of_gradients[l] - sums_of_gradients[c]
                offset = triangular_number(l)

                for c2 in range(c + 1):
                    l2 = offset + get_index(c2, label_indices)
                    hessians[i] = total_sums_of_hessians[l2] - sums_of_hessians[i]
                    i += 1
        else:
            gradients = sums_of_gradients
            hessians = sums_of_hessians

        # Calculate the optimal scores by solving a system of linear equations...
        cdef float64[::1] scores = dsysv_float64(hessians, gradients, l2_regularization_weight)
        prediction.predicted_scores = scores

        # Calculate overall quality score as (gradients * scores) + (0.5 * (scores * (hessians * scores)))...
        cdef float64 overall_quality_score = -ddot_float64(scores, gradients)
        cdef float64[::1] tmp = dspmv_float64(hessians, scores)
        overall_quality_score += 0.5 * ddot_float64(scores, tmp)
        overall_quality_score += 0.5 * l2_regularization_weight * l2_norm_pow(scores)
        prediction.overall_quality_score = overall_quality_score
        return prediction

    cdef apply_predictions(self, intp[::1] covered_example_indices, intp[::1] label_indices,
                           float64[::1] predicted_scores):
        cdef intp num_predicted_labels = predicted_scores.shape[0]
        cdef uint8[::1, :] ground_truth = self.ground_truth
        cdef float64[::1, :] current_scores = self.current_scores
        cdef float64[::1, :] gradients = self.gradients
        cdef intp num_labels = gradients.shape[1]
        cdef float64[::1] total_sums_of_gradients = self.total_sums_of_gradients
        cdef float64[::1, :] hessians = self.hessians
        cdef float64[::1] total_sums_of_hessians = self.total_sums_of_hessians
        cdef float64[::1] exponentials = array_float64(num_labels)
        cdef float64[::1] expected_scores = array_float64(num_labels)
        cdef float64 expected_score, exponential, score, sum_of_exponentials, sum_of_exponentials_pow
        cdef intp r, c, c2, l, i

        # Only the examples that are covered by the new rule must be considered...
        for r in covered_example_indices:
            # Traverse the labels for which the new rule predicts to update the currently predicted scores...
            for c in range(num_predicted_labels):
                l = get_index(c, label_indices)
                current_scores[r, l] += predicted_scores[c]

            # Traverse the labels of the current example to create arrays of expected scores and exponentials that are
            # shared among the upcoming calculations of gradients and hessians...
            sum_of_exponentials = 1

            for c in range(num_labels):
                expected_score = convert_label_into_score(ground_truth[r, c])
                expected_scores[c] = expected_score
                exponential = exp(-expected_score * current_scores[r, c])
                exponentials[c] = exponential
                sum_of_exponentials += exponential

            sum_of_exponentials_pow = pow(sum_of_exponentials, 2)

            # Traverse the labels again to update the gradients and hessians...
            i = 0

            for c in range(num_labels):
                expected_score = expected_scores[c]
                exponential = exponentials[c]
                score = current_scores[r, c]

                # Calculate the first derivative (gradient) of the loss function with respect to the current label and
                # add it to the matrix of gradients...
                tmp = gradients[r, c]
                total_sums_of_gradients[c] -= tmp
                tmp = (expected_score * exponential) / sum_of_exponentials
                # Note: The sign of the gradient is inverted (from negative to positive), because otherwise, when using
                # the sums of gradients as the ordinates for solving a system of linear equations in the function
                # `evaluate_label_dependent_predictions`, the sign must be inverted again...
                gradients[r, c] = tmp
                total_sums_of_gradients[c] += tmp

                # Calculate the second derivatives (hessians) of the loss function with respect to the current label and
                # each of the other labels and add them to the matrix of hessians...
                for c2 in range(c):
                    tmp = hessians[r, i]
                    total_sums_of_hessians[i] -= tmp
                    tmp = exp(-expected_scores[c2] * current_scores[r, c2] - expected_score * score)
                    tmp = (expected_scores[c2] * expected_score * tmp) / sum_of_exponentials_pow
                    hessians[r, i] = -tmp
                    total_sums_of_hessians[i] -= tmp
                    i += 1

                # Calculate the second derivative (hessian) of the loss function with respect to the current label and
                # add it to the matrix of hessians...
                tmp = hessians[r, i]
                total_sums_of_hessians[i] -= tmp
                tmp = (pow(expected_score, 2) * exponential * (sum_of_exponentials - exponential)) / sum_of_exponentials_pow
                hessians[r, i] = tmp
                total_sums_of_hessians[i] += tmp
                i += 1
