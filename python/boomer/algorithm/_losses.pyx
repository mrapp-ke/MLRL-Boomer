# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides classes that implement different loss functions to be minimized during training.
"""
from boomer.algorithm._arrays cimport array_float64, matrix_float64
from boomer.algorithm._utils cimport get_index, convert_label_into_score
from boomer.algorithm._math cimport divide_or_zero_float64, triangular_number, dsysv_float64

from libc.math cimport pow, exp


cdef class Prediction:
    """
    Assess the overall quality of a rule's predictions for one or several labels.
    """

    def __cinit__(self, float64[::1] predicted_scores, float64 overall_quality_score):
        """
        :param predicted_scores:    An array of dtype float, shape `(num_predicted_labels)`, representing the scores
                                    that are predicted by the rule
        :param quality_score:       The overall quality score
        """
        self.predicted_scores = predicted_scores
        self.overall_quality_score = overall_quality_score


cdef class LabelIndependentPrediction(Prediction):
    """
    Assesses the quality of a rule's predictions for one or several labels independently from each other.
    """

    def __cinit__(self, float64[::1] predicted_scores, float64[::1] quality_scores, float64 overall_quality_score):
        """
        :param predicted_scores:        An array of dtype float, shape `(num_predicted_labels)`, representing the scores
                                        that are predicted by the rule
        :param quality_scores:          An array of dtype float, shape `(num_predicted_labels)`, representing the
                                        quality scores for the individual labels
        :param overall_quality_score:   The overall quality score
        """
        self.predicted_scores = predicted_scores
        self.quality_scores = quality_scores
        self.overall_quality_score = overall_quality_score


cdef class Loss:
    """
    A base class for all decomposable or non-decomposable loss functions. A loss function can be used to calculate the
    optimal scores to be predicted by rules for individual labels and covered examples.
    """

    cdef float64[::1] calculate_default_scores(self, uint8[::1, :] y):
        """
        Calculates the optimal scores to be predicted by the default rule for each label and example.

        This function must be called prior to calling any other function provided by this class. It calculates and
        caches the gradients (and hessians in case of a non-decomposable loss function) based on the expected confidence
        scores and the scores predicted by the default rule.

        Furthermore, this function also computes and caches an array storing the total sum of gradients for each label.
        This is necessary to later be able to search for the optimal scores to be predicted by rules that cover all
        examples provided to the search, as well as by rules that do not cover these instances (but all other ones) at
        the same time instead of requiring two passes through the examples. When using instance sub-sampling, before
        invoking the function `begin_search`, the function `begin_instance_sub_sampling` must be called, followed by
        invocations of the function `update_sub_sample` for each of the examples contained in the sample.

        :param y:   An array of dtype float, shape `(num_examples, num_labels)`, representing the labels of the training
                    examples according to the ground truth
        :return:    An array of dtype float, shape `(num_labels)`, representing the optimal scores to be predicted by
                    the default rule for each label and example
        """
        pass

    cdef begin_instance_sub_sampling(self):
        """
        Resets the cached sum of gradients (and hessians in case of a non-decomposable loss function) for each label to
        0.

        This function must be invoked before the functions `begin_instance_sub_sampling` and `begin_search` if any type
        of instance sub-sampling, e.g. bagging, is used.
        """
        pass

    cdef update_sub_sample(self, intp example_index):
        """
        Updates the total sum of gradients (and hessians in case of a non-decomposable loss function) for each label
        based on an example that has been chosen to be included in the sub-sample.

        This function must be invoked for each example included in the sample after the function
        `begin_instance_sub_sampling' and before `begin_search`.

        :param example_index: The index of an example that has been chosen to be included in the sample
        """
        pass

    cdef begin_search(self, intp[::1] label_indices):
        """
        Begins a new search to find the optimal scores to be predicted by candidate rules for individual labels.

        This function must be called prior to searching for the best refinement with respect to a certain attribute in
        order to reset the sums of gradients (and hessians in case of a non-decomposable loss function) cached
        internally to calculate the optimal scores more efficiently. Subsequent invocations of the function
        `update_search` can be used to update the cached values afterwards based on a single, newly covered example.
        Invoking the function `evaluate_label_independent_predictions` or `evaluate_label_dependent_predictions` at any
        point of the search (`update_search` must be called at least once before!) will yield the optimal scores to be
        predicted by a rule that covers all examples given so far, as well as corresponding quality scores that measure
        the quality of such a rule's predictions.

        When a new rule has been induced, the function `apply_predictions` must be invoked in order to update the cached
        gradients (and hessians in case of a non-decomposable loss function).

        :param label_indices: An array of dtype int, shape `(num_predicted_labels)`, representing the indices of the
                              labels for which the rule should predict or None, if the rule may predict for all labels
        """
        pass

    cdef update_search(self, intp example_index, uint32 weight):
        """
        Updates the cached sums of gradients (and hessians in case of a non-decomposable loss function) based on a
        single, newly covered example.

        Subsequent invocations of the function `evaluate_label_independent_predictions` or
        `evaluate_label_dependent_predictions` will yield the optimal scores to be predicted by a rule that covers all
        examples given so far, as well as corresponding quality scores that measure the quality of such a rule's
        predictions.

        :param example_index:   The index of the newly-covered example in the entire training data set
        :param weight:          The weight of the newly covered example
        """
        pass

    cdef LabelIndependentPrediction evaluate_label_independent_predictions(self, bint uncovered):
        """
        Calculates the optimal scores to be predicted by a rule that covers all examples that have been provided so far
        via the function `update_search`, respectively by a rule that covers all examples that have not been provided
        yet. Additionally, quality scores that measure the quality of the predicted scores are calculated for each
        label.

        The optimal score to be predicted for an individual label is calculated independently from the other labels. In
        case of a non-decomposable loss function, it is assumed that the rule will abstain, i.e., predict 0, for the
        other labels. The same assumption is used to calculate a quality score for each label independently.

        The calculated scores correspond to the label indices provided to the `begin_search` function. If no label
        indices were provided, scores for all labels are calculated.

        :param uncovered:   0, if the scores for a rule that covers all examples that have been provided so far should
                            be calculated, 1, if the scores for a rule that covers all examples that have not been
                            provided yet should be calculated
        :return:            A `LabelIndependentPrediction` that stores the optimal scores to be predicted by the rule,
                            as well as the corresponding quality scores for each label
        """
        pass

    cdef Prediction evaluate_label_dependent_predictions(self, bint uncovered):
        """
        Calculates the optimal scores to be predicted by a rule that covers all examples that have been provided so far
        via the function `update_search`, respectively by a rule that covers all examples that have not been provided
        yet. Additionally, a single quality score that measures the quality of the predicted scores is calculated.

        The optimal score to be predicted for an individual label is calculated with respect to the predictions for the
        other labels. In case of a decomposable loss function, i.e., if the labels are independent from each other, the
        optimal scores provided by the function `evaluate_label_independent_predictions` are the same.

        The calculated scores correspond to the label indices provided to the `begin_search` function. If no label
        indices were provided, scores for all labels are calculated.

        :param uncovered:   0, if the scores for a rule that covers all examples that have been provided so far should
                            be calculated, 1, if the scores for a rule that covers all examples that have not been
                            provided yet should be calculated
        :return:            A `Prediction` that stores the optimal scores to be predicted by the rule, as well as its
                            overall quality score
        """
        pass

    cdef apply_predictions(self, intp[::1] covered_example_indices, intp[::1] label_indices,
                           float64[::1] predicted_scores):
        """
        Updates the cached gradients (and hessians in case of a non-decomposable loss function) based on the predictions
        provided by newly induced rules.

        :param covered_example_indices: An array of dtype int, shape `(num_covered_examples)`, representing the indices
                                        of the examples that are covered by the newly induced rule, regardless of
                                        whether they are contained in the sub-sample or not
        :param label_indices:           An array of dtype int, shape `(num_predicted_labels)`, representing the indices
                                        of the labels for which the newly induced rule predicts or None, if the rule
                                        predicts for all labels
        :param predicted_scores:        An array of dtype float, shape `(num_predicted_labels)`, representing the scores
                                        that are predicted by the newly induced rule
        """
        pass

cdef class DecomposableLoss(Loss):
    """
    A base class for all decomposable loss functions.
    """

    cdef float64[::1] calculate_default_scores(self, uint8[::1, :] y):
        pass

    cdef begin_instance_sub_sampling(self):
        pass

    cdef update_sub_sample(self, intp example_index):
        pass

    cdef begin_search(self, intp[::1] label_indices):
        pass

    cdef update_search(self, intp example_index, uint32 weight):
        pass

    cdef LabelIndependentPrediction evaluate_label_independent_predictions(self, bint uncovered):
        pass

    cdef Prediction evaluate_label_dependent_predictions(self, bint uncovered):
        # In case of a decomposable loss, the label-dependent predictions are the same as the label-independent
        # predictions...
        return self.evaluate_label_independent_predictions(uncovered)

    cdef apply_predictions(self, intp[::1] covered_example_indices, intp[::1] label_indices,
                           float64[::1] predicted_scores):
        pass


cdef class NonDecomposableLoss(Loss):
    """
    A base class for all non-decomposable loss functions.
    """

    cdef float64[::1] calculate_default_scores(self, uint8[::1, :] y):
        pass

    cdef begin_instance_sub_sampling(self):
        pass

    cdef update_sub_sample(self, intp example_index):
        pass

    cdef begin_search(self, intp[::1] label_indices):
        pass

    cdef update_search(self, intp example_index, uint32 weight):
        pass

    cdef LabelIndependentPrediction evaluate_label_independent_predictions(self, bint uncovered):
        pass

    cdef Prediction evaluate_label_dependent_predictions(self, bint uncovered):
        pass

    cdef apply_predictions(self, intp[::1] covered_example_indices, intp[::1] label_indices,
                           float64[::1] predicted_scores):
        pass


cdef class SquaredErrorLoss(DecomposableLoss):
    """
    A multi-label variant of the squared error loss.
    """

    cdef float64[::1] calculate_default_scores(self, uint8[::1, :] y):
        cdef intp num_rows = y.shape[0]
        cdef intp num_cols = y.shape[1]
        cdef float64[::1, :] gradients = matrix_float64(num_rows, num_cols)
        cdef float64[::1] total_sums_of_gradients = array_float64(num_cols)
        cdef float64[::1] scores = array_float64(num_cols)
        cdef float64 sum_of_hessians = 2 * num_rows
        cdef float64 sum_of_gradients, expected_score, score, gradient
        cdef intp r, c

        for c in range(num_cols):
            # Column-wise sum up gradients for the current label...
            sum_of_gradients = 0

            for r in range(num_rows):
                expected_score = 2 * convert_label_into_score(y[r, c])
                # Note: As the matrix of gradients is unused at this point, we use it for caching the expected scores
                # instead of allocating a new array. The values in the matrix are overwritten later on with the actual
                # gradients.
                gradients[r, c] = expected_score
                sum_of_gradients -= expected_score

            # Calculate optimal score to be predicted by the default rule for the current label...
            score = -sum_of_gradients / sum_of_hessians
            scores[c] = score

            # Traverse column again to calculate updated gradients based on the calculated score...
            sum_of_gradients = 0
            score = 2 * score

            for r in range(num_rows):
                gradient = score - gradients[r, c]
                gradients[r, c] = gradient
                sum_of_gradients += gradient

            total_sums_of_gradients[c] = sum_of_gradients

        # Cache the matrix of gradients...
        self.gradients = gradients

        # Cache the total sums of gradients for each label...
        self.total_sums_of_gradients = total_sums_of_gradients

        # Cache total sum of hessians...
        self.total_sum_of_hessians = sum_of_hessians

        return scores

    cdef begin_instance_sub_sampling(self):
        # Reset total sum of hessians to 0...
        self.total_sum_of_hessians = 0

        # Reset total sums of gradients to 0...
        cdef float64[::1] total_sums_of_gradients = self.total_sums_of_gradients
        total_sums_of_gradients[:] = 0

    cdef update_sub_sample(self, intp example_index):
        # Update total sum of hessians...
        cdef float64 total_sum_of_hessians = self.total_sum_of_hessians
        total_sum_of_hessians += 2
        self.total_sum_of_hessians = total_sum_of_hessians

        # Update total sums of gradients...
        cdef float64[::1, :] gradients = self.gradients
        cdef float64[::1] total_sums_of_gradients = self.total_sums_of_gradients
        cdef intp num_cols = gradients.shape[1]
        cdef intp c

        for c in range(num_cols):
            total_sums_of_gradients[c] += gradients[example_index, c]

    cdef begin_search(self, intp[::1] label_indices):
        # Determine the number of labels considered by the upcoming search...
        cdef intp num_labels
        cdef float64[::1, :] gradients

        if label_indices is None:
            gradients = self.gradients
            num_labels = gradients.shape[1]
        else:
            num_labels = label_indices.shape[0]

        # Reset sums of gradients to 0...
        cdef float64[::1] sums_of_gradients = array_float64(num_labels)
        sums_of_gradients[:] = 0
        self.sums_of_gradients = sums_of_gradients

        # Reset sum of hessians to 0...
        cdef float64 sum_of_hessians = 0
        self.sum_of_hessians = sum_of_hessians

        # Cache the given label indices...
        self.label_indices = label_indices

        # Initialize object for storing potential predictions once to avoid object-recreation at each update...
        cdef float64[::1] predicted_scores = array_float64(num_labels)
        cdef float64[::1] quality_scores = array_float64(num_labels)
        cdef LabelIndependentPrediction prediction = LabelIndependentPrediction(predicted_scores, quality_scores, 0)
        self.prediction = prediction

    cdef update_search(self, intp example_index, uint32 weight):
        # Update sum of hessians...
        cdef float64 sum_of_hessians = self.sum_of_hessians
        sum_of_hessians += (weight * 2)
        self.sum_of_hessians = sum_of_hessians

        # Column-wise update sums of gradients...
        cdef float64[::1, :] gradients = self.gradients
        cdef float64[::1] sums_of_gradients = self.sums_of_gradients
        cdef intp num_labels = sums_of_gradients.shape[0]
        cdef intp[::1] label_indices = self.label_indices
        cdef intp c, l

        for c in range(num_labels):
            l = get_index(c, label_indices)
            sums_of_gradients[c] += (weight * gradients[example_index, l])

    cdef LabelIndependentPrediction evaluate_label_independent_predictions(self, bint uncovered):
        cdef float64[::1] sums_of_gradients = self.sums_of_gradients
        cdef float64 sum_of_hessians = self.sum_of_hessians
        cdef intp num_labels = sums_of_gradients.shape[0]
        cdef LabelIndependentPrediction prediction = self.prediction
        cdef float64[::1] predicted_scores = prediction.predicted_scores
        cdef float64[::1] quality_scores = prediction.quality_scores
        cdef float64 sum_of_gradients, score, score_halved, overall_quality_score
        cdef float64[::1] total_sums_of_gradients
        cdef intp[::1] label_indices
        cdef intp c, l

        if uncovered:
            sum_of_hessians = self.total_sum_of_hessians - sum_of_hessians
            total_sums_of_gradients = self.total_sums_of_gradients
            label_indices = self.label_indices

        overall_quality_score = 0

        for c in range(num_labels):
            sum_of_gradients = sums_of_gradients[c]

            if uncovered:
                l = get_index(c, label_indices)
                sum_of_gradients = total_sums_of_gradients[l] - sum_of_gradients

            # Calculate predicted score...
            score = divide_or_zero_float64(-sum_of_gradients, sum_of_hessians)
            predicted_scores[c] = score

            # Calculate quality score...
            score_halved = score / 2
            score = (sum_of_gradients * score) + (score_halved * sum_of_hessians * score)
            quality_scores[c] = score
            overall_quality_score += score

        prediction.overall_quality_score = overall_quality_score
        return prediction

    cdef apply_predictions(self, intp[::1] covered_example_indices, intp[::1] label_indices,
                           float64[::1] predicted_scores):
        cdef intp num_labels = predicted_scores.shape[0]
        cdef float64[::1, :] gradients = self.gradients
        cdef float64[::1] total_sums_of_gradients = self.total_sums_of_gradients
        cdef float64 total_sum_of_gradients, predicted_score, gradient
        cdef intp c, l, i

        # Only the labels that are predicted by the new rule must be considered...
        for c in range(num_labels):
            l = get_index(c, label_indices)
            predicted_score = 2 * predicted_scores[c]
            total_sum_of_gradients = total_sums_of_gradients[l]

            # Only the examples that are covered by the new rule must be considered...
            for i in covered_example_indices:
                gradient = gradients[i, l]

                # Update the total sum of gradients by subtracting the old gradient and adding the new one...
                # If instance sub-sampling is used, this will cause the total sum of gradients to become incorrect.
                # However, this doesn't matter, because it will be recalculated when re-sampling for learning the next
                # rule anyway.
                total_sum_of_gradients -= gradient
                gradient += predicted_score
                total_sum_of_gradients += gradient

                # Update the gradient for the current label and example...
                gradients[i, l] = gradient

            # Update the sum of gradients for the current label...
            total_sums_of_gradients[l] = total_sum_of_gradients


cdef class LogisticLoss(NonDecomposableLoss):
    """
    A multi-label variant of the logistic loss that is applied example-wise.
    """

    cdef float64[::1] calculate_default_scores(self, uint8[::1, :] y):
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
        # efficiency, we store the hessians in an one-dimensional array by appending the rows of the matrix of
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

                # Calculate the second derivative (hessian) of the loss function with respect to the current label and
                # add it to the array of coefficients...
                coefficients[i] += (pow(expected_score, 2) * num_cols) / sum_of_exponentials_pow

                # Calculate the second derivatives (hessians) of the loss function with respect to the current label and
                # each of the other labels and add them to the array of coefficients...
                for c2 in range(c + 1, num_cols):
                    i += 1
                    coefficients[i] -= (expected_score * expected_scores[c2]) / sum_of_exponentials_pow

                i += 1

        # Compute the optimal scores to be predicted by the default rule by solving the system of linear equations...
        cdef float64[::1] scores = dsysv_float64(coefficients, ordinates)

        # We must traverse each example again to calculate the updated gradients and hessians based on the calculated
        # scores...
        cdef float64[::1] exponentials = array_float64(num_cols)
        cdef float64[::1, :] gradients = matrix_float64(num_rows, num_cols)
        cdef float64[::1] total_sums_of_gradients = array_float64(num_cols)
        cdef float64[::1, :] hessians = matrix_float64(num_rows, num_hessians)
        cdef float64[::1] total_sums_of_hessians = array_float64(num_hessians)
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

                # Calculate the first derivative (gradient) of the loss function with respect to the current label and
                # add it to the matrix of gradients...
                tmp = (expected_score * exponential) / sum_of_exponentials
                gradients[r, c] = -tmp
                total_sums_of_gradients[c] -= tmp

                # Calculate the second derivative (hessian) of the loss function with respect to the current label and
                # add it to the matrix of hessians...
                tmp = (pow(expected_score, 2) * exponential * (sum_of_exponentials - exponential)) / sum_of_exponentials_pow
                hessians[r, i] = tmp
                total_sums_of_hessians[i] += tmp

                # Calculate the second derivatives (hessians) of the loss function with respect to the current label and
                # each of the other labels and add them to the matrix of hessians...
                for c2 in range(c + 1, num_cols):
                    i += 1
                    exponential = exp(-expected_score * score - expected_scores[c2] * scores[c2])
                    tmp = (expected_score * expected_scores[c2] * exponential) / sum_of_exponentials_pow
                    hessians[r, i] = -tmp
                    total_sums_of_hessians[i] -= tmp

                i += 1

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
        # TODO Allocate the data structure for storing the predicted scores and quality scores of rules

    cdef update_search(self, intp example_index, uint32 weight):
        cdef float64[::1, :] gradients = self.gradients
        cdef float64[::1, :] hessians = self.hessians
        cdef intp total_num_gradients = gradients.shape[1]
        cdef intp total_num_hessians = hessians.shape[1]
        cdef float64[::1] sums_of_gradients = self.sums_of_gradients
        cdef float64[::1] sums_of_hessians = self.sums_of_hessians
        cdef intp num_gradients = sums_of_gradients.shape[0]
        cdef intp num_hessians = sums_of_hessians.shape[0]
        cdef intp[::1] label_indices = self.label_indices
        cdef intp c, c2, l, l2, i, offset
        i = 0

        for c in range(num_gradients):
            l = get_index(c, label_indices)
            sums_of_gradients[c] += (weight * gradients[example_index, l])
            offset = total_num_hessians - triangular_number(total_num_gradients - l)

            for c2 in range(c, num_gradients):
                l2 = offset + get_index(c2, label_indices) - l
                sums_of_hessians[i] += (weight * hessians[example_index, l2])
                i += 1

    cdef LabelIndependentPrediction evaluate_label_independent_predictions(self, bint uncovered):
        # TODO
        pass

    cdef Prediction evaluate_label_dependent_predictions(self, bint uncovered):
        # TODO
        pass

    cdef apply_predictions(self, intp[::1] covered_example_indices, intp[::1] label_indices,
                           float64[::1] predicted_scores):
        # TODO
        pass
