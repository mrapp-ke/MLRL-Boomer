# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=False
from boomer.algorithm._utils cimport get_weight

from cython.view cimport array as cvarray
from libc.math cimport pow

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
        Invoking the function `calculate_scores` at any point of the search (`update_search` must be called at least
        once before!) will yield the optimal scores to be predicted by a rule that covers all examples given so far.
        Accordingly, the function `calculate_quality_score` allows to retrieve scores for each label that measure the
        quality of such a rule's predictions.

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

        Subsequent invocations of the function `calculate_scores` will yield the optimal scores to predicted by a rule
        that covers all examples given so far. Accordingly, the function `calculate_quality_score` allows to retrieve
        scores for each label that measure the quality of such a rule's predictions.

        :param example_index:   The index of the newly-covered example in the entire training data set
        :param weight:          The weight of the newly covered example
        """
        pass

    cdef float64[::1, :] calculate_predicted_and_quality_scores(self):
        """
        Calculates the optimal scores to be predicted by rules that cover all examples that have been provided so far
        via the function `update_search`, respectively that cover all examples that have not been provided yet, as well
        as corresponding quality scores that measure the quality of the predicted scores.

        The calculated scores correspond to the label indices provided to the `begin_search` function. If no label
        indices were provided, scores for all labels are calculated.

        :return: An array of dtype float, shape `(4, num_predicted_labels)`, representing the optimal scores to be
                 predicted by a rule that covers all examples provided so far in the 1st row, as well as the
                 corresponding quality scores in the 2nd row. Accordingly, the 3rd row represents the optimal scores to
                 be predicted by a rule that covers all examples that have not been not provided yet, and the 4th row
                 contains the corresponding quality scores.
        """
        pass

    cdef float64 calculate_quality_score(self, float64[::1] predicted_scores):
        """
        Calculates an overall quality score that measures the quality of the scores that are predicted by a rule for one
        or several labels of the examples that have been provided so far via the function `update_search`.

        This function may be used instead of the function `calculate_predicted_and_quality_scores` to evaluate the
        quality of a previously induced rule based on new examples that have not been used to grow the rule.

        The given predicted scores must correspond to the label indices provided to the `begin_search` function. If no
        label indices were provided, predicted scores for all labels must be provided.

        :param predicted_scores:    An array of dtype float, shape `(num_predicted_labels)`, representing the scores
                                    that are predicted by the rule
        :return:                    An overall quality score that measures the quality of the rule
        """
        pass

    cdef apply_predictions(self, intp[::1] covered_example_indices, uint32[::1] weights, intp[::1] label_indices,
                           float64[::1] predicted_scores):
        """
        Updates the cached gradients (and hessians in case of a non-decomposable loss function) based on the predictions
        provided by newly induced rules.

        :param covered_example_indices: An array of dtype int, shape `(num_covered_examples)`, representing the indices
                                        of the examples that are covered by the newly induced rule, regardless of
                                        whether they are contained in the sub-sample or not
        :param weights:                 An array of dtype int, shape `(num_examples)`, representing the weights of all
                                        training examples
        :param label_indices:           An array of dtype int, shape `(num_predicted_labels)`, representing the indices
                                        of the labels for which the newly induced rule predicts
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

    cdef float64[::1, :] calculate_predicted_and_quality_scores(self):
        pass

    cdef float64 calculate_quality_score(self, float64[::1] predicted_scores):
        pass

    cdef apply_predictions(self, intp[::1] covered_example_indices, uint32[::1] weights, intp[::1] label_indices,
                           float64[::1] predicted_scores):
        pass


cdef class SquaredErrorLoss(DecomposableLoss):
    """
    A multi-label variant of the squared error loss.
    """

    cdef float64[::1] calculate_default_scores(self, uint8[::1, :] y):
        cdef intp num_rows = y.shape[0]
        cdef intp num_cols = y.shape[1]
        cdef float64[::1, :] gradients = cvarray(shape=(num_rows, num_cols), itemsize=sizeof(float64), format='d',
                                                 mode='fortran')
        cdef float64[::1] total_sums_of_gradients = cvarray(shape=(num_cols,), itemsize=sizeof(float64), format='d',
                                                            mode='c')
        cdef float64[::1] scores = cvarray(shape=(num_cols,), itemsize=sizeof(float64), format='d', mode='c')
        cdef float64 sum_of_hessians = 2 * num_rows
        cdef float64 sum_of_gradients, expected_score, score, gradient
        cdef float64[::1] expected_scores = cvarray(shape=(num_rows,), itemsize=sizeof(float64), format='d', mode='c')
        cdef intp r, c

        for c in range(num_cols):
            # Column-wise sum up gradients for the current label...
            sum_of_gradients = 0

            for r in range(num_rows):
                expected_score = 2 * __convert_label_into_score(y[r, c])
                expected_scores[r] = expected_score
                sum_of_gradients -= expected_score

            # Calculate optimal score to be predicted by the default rule for the current label...
            score = -sum_of_gradients / sum_of_hessians
            scores[c] = score

            # Traverse column again to calculate updated gradients based on the calculated score...
            sum_of_gradients = 0
            score = 2 * score

            for r in range(num_rows):
                gradient = score - expected_scores[r]
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
        cdef float64 total_sum_of_hessians = 0
        self.total_sum_of_hessians = total_sum_of_hessians

        # Reset total sums of gradients to 0...
        cdef float64[::1, :] gradients = self.gradients
        cdef intp num_cols = gradients.shape[1]
        cdef float64[::1] total_sums_of_gradients = cvarray(shape=(num_cols,), itemsize=sizeof(float64), format='d',
                                                            mode='c')
        total_sums_of_gradients[:] = 0
        self.total_sums_of_gradients = total_sums_of_gradients

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
        # Reset sum of hessians to 0...
        cdef float64 sum_of_hessians = 0
        self.sum_of_hessians = sum_of_hessians

        # Reset sums of gradients to 0...
        cdef intp num_labels
        cdef float64[::1, :] gradients

        if label_indices is None:
            gradients = self.gradients
            num_labels = gradients.shape[1]
        else:
            num_labels = label_indices.shape[0]

        cdef float64[::1] sums_of_gradients = cvarray(shape=(num_labels,), itemsize=sizeof(float64), format='d',
                                                      mode='c')
        sums_of_gradients[:] = 0
        self.sums_of_gradients = sums_of_gradients
        self.label_indices = label_indices

        # Initialize array of scores once to avoid array-recreation at each update...
        cdef float64[::1, :] predicted_and_quality_scores = cvarray(shape=(4, num_labels), itemsize=sizeof(float64),
                                                                    format='d', mode='fortran')
        self.predicted_and_quality_scores = predicted_and_quality_scores


    cdef update_search(self, intp example_index, uint32 weight):
        # Update sum of hessians...
        cdef float64 sum_of_hessians = self.sum_of_hessians
        sum_of_hessians += weight * 2
        self.sum_of_hessians = sum_of_hessians

        # Column-wise update sums of gradients...
        cdef float64[::1, :] gradients = self.gradients
        cdef float64[::1] sums_of_gradients = self.sums_of_gradients
        cdef intp num_labels = sums_of_gradients.shape[0]
        cdef intp[::1] label_indices = self.label_indices
        cdef intp c, l

        for c in range(num_labels):
            l = __get_label_index(c, label_indices)
            sums_of_gradients[c] += weight * gradients[example_index, l]

    cdef float64[::1, :] calculate_predicted_and_quality_scores(self):
        cdef float64[::1, :] predicted_and_quality_scores = self.predicted_and_quality_scores
        cdef float64[::1] sums_of_gradients = self.sums_of_gradients
        cdef float64[::1] total_sums_of_gradients = self.total_sums_of_gradients
        cdef float64 sum_of_hessians = self.sum_of_hessians
        cdef float64 total_sum_of_hessians = self.total_sum_of_hessians
        cdef float64 sum_of_hessians_uncovered = total_sum_of_hessians - sum_of_hessians
        cdef intp num_labels = sums_of_gradients.shape[0]
        cdef intp[::1] label_indices = self.label_indices
        cdef float64 sum_of_gradients, score, score_halved
        cdef intp c, l

        for c in range(num_labels):
            sum_of_gradients = sums_of_gradients[c]

            # Calculate score to be predicted by a rule that covers the examples that have been provided so far...
            score = -sum_of_gradients / sum_of_hessians
            predicted_and_quality_scores[0, c] = score

            # Calculate quality score for a rule that covers the examples that have been provided so far..
            score_halved = score / 2
            predicted_and_quality_scores[1, c] = (sum_of_gradients * score) + (score_halved * sum_of_hessians * score)

            # Calculate score to be predicted by a rule that covers the examples that have not been provided yet...
            l = __get_label_index(c, label_indices)
            score = -(total_sums_of_gradients[l] - sum_of_gradients) / sum_of_hessians_uncovered
            predicted_and_quality_scores[2, c] = score

            # Calculate quality score for a rule that covers the examples that have not been provided yet...
            score_halved = score / 2
            predicted_and_quality_scores[3, c] = ((total_sums_of_gradients[l] - sum_of_gradients) * score) + (score_halved * sum_of_hessians_uncovered * score)

        return predicted_and_quality_scores

    cdef float64 calculate_quality_score(self, float64[::1] predicted_scores):
        cdef float64 quality_score = 0
        cdef float64[::1] sums_of_gradients = self.sums_of_gradients
        cdef float64 sum_of_hessians = self.sum_of_hessians
        cdef intp num_labels = sums_of_gradients.shape[0]
        cdef float64 score, score_halved
        cdef intp c

        # For each of the labels, calculate a quality score based on the score that is predicted by the rule for the
        # respective label and the examples that have been provided so far and add it to the overall quality score...
        for c in range(num_labels):
            sum_of_gradients = sums_of_gradients[c]
            score = predicted_scores[c]
            score_halved = score / 2
            quality_score += (sum_of_gradients * score) + (score_halved * sum_of_hessians * score)

        return quality_score

    cdef apply_predictions(self, intp[::1] covered_example_indices, uint32[::1] weights, intp[::1] label_indices,
                           float64[::1] predicted_scores):
        cdef intp num_labels = label_indices.shape[0]
        cdef float64[::1, :] gradients = self.gradients
        cdef float64[::1] total_sums_of_gradients = self.total_sums_of_gradients
        cdef float64 total_sum_of_gradients, predicted_score, old_gradient, new_gradient
        cdef uint32 weight
        cdef intp c, l, i

        # Only the labels that are predicted by the new rule must be considered...
        for c in range(num_labels):
            l = label_indices[c]
            predicted_score = 2 * predicted_scores[c]
            total_sum_of_gradients = total_sums_of_gradients[l]

            # Only the examples that are covered by the new rule must be considered...
            for i in covered_example_indices:
                # Update the old gradient by adding the predicted score...
                old_gradient = gradients[i, l]
                new_gradient = old_gradient + predicted_score
                gradients[i, l] = new_gradient

                # If the current example is contained in the sub-sample, update the total sum of gradients by
                # subtracting the old gradient and adding the new one...
                weight = get_weight(i, weights)

                if weight > 0:
                    total_sum_of_gradients = total_sum_of_gradients - old_gradient + new_gradient

            # Update the sum of gradients for the current label...
            total_sums_of_gradients[l] = total_sum_of_gradients
