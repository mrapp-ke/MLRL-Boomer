# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=False
from boomer.algorithm._utils cimport divide

from cython.view cimport array as cvarray
from libc.math cimport pow, exp

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
        Invoking the function `calculate_predicted_and_quality_scores` at any point of the search (`update_search` must
        be called at least once before!) will yield the optimal scores to be predicted by a rule that covers all
        examples given so far, as well as corresponding quality scores that measure the quality of such a rule's
        predictions.

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

        Subsequent invocations of the function `calculate_predicted_and_quality_scores` will yield the optimal scores to
        be predicted by a rule that covers all examples given so far, as well as corresponding quality scores that
        measure the quality of such a rule's predictions.

        :param example_index:   The index of the newly-covered example in the entire training data set
        :param weight:          The weight of the newly covered example
        """
        pass

    cdef float64[::1, :] calculate_predicted_and_quality_scores(self, bint include_uncovered):
        """
        Calculates the optimal scores to be predicted by a rule that covers all examples that have been provided so far
        via the function `update_search`, as well as (optionally) by a rule that covers all examples that have not been
        provided yet. Additionally, for each label, quality scores that measure the quality of the predicted scores are
        calculated.

        The calculated scores correspond to the label indices provided to the `begin_search` function. If no label
        indices were provided, scores for all labels are calculated.

        :param include_uncovered:   1, if the scores for a rule that covers all examples that have not been provided
                                    should be calculated, 0 otherwise
        :return:                    An array of dtype float, shape `(4, num_predicted_labels)`, representing the optimal
                                    scores to be predicted by a rule that covers all examples provided so far in the 1st
                                    row, as well as the corresponding quality scores in the 2nd row. Accordingly, the
                                    3rd row represents the optimal scores to be predicted by a rule that covers all
                                    examples that have not been not provided yet, and the 4th row contains the
                                    corresponding quality scores. If `include_uncovered` is 0, the elements in the 3rd
                                    and 4th row are undefined
        """
        pass

    cdef float64[::1] calculate_predicted_scores(self):
        """
        Calculates the scores to be predicted by a rule that covers all examples that have been provided so far via the
        function `update_search`.

        The calculated scores correspond to the label indices provided to the `begin_search` function. If no label
        indices were provided, scores for all labels are calculated.

        :return: An array of dtype float, shape `(num_predicted_labels)`, representing the optimal scores to be
                 predicted by a rule that covers all examples provided so far
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

    cdef float64[::1, :] calculate_predicted_and_quality_scores(self, bint include_uncovered):
        pass

    cdef float64[::1] calculate_predicted_scores(self):
        pass

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

    cdef float64[::1, :] calculate_predicted_and_quality_scores(self, bint include_uncovered):
        pass

    cdef float64[::1] calculate_predicted_scores(self):
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
        sum_of_hessians += (weight * 2)
        self.sum_of_hessians = sum_of_hessians

        # Column-wise update sums of gradients...
        cdef float64[::1, :] gradients = self.gradients
        cdef float64[::1] sums_of_gradients = self.sums_of_gradients
        cdef intp num_labels = sums_of_gradients.shape[0]
        cdef intp[::1] label_indices = self.label_indices
        cdef intp c, l

        for c in range(num_labels):
            l = __get_label_index(c, label_indices)
            sums_of_gradients[c] += (weight * gradients[example_index, l])

    cdef float64[::1, :] calculate_predicted_and_quality_scores(self, bint include_uncovered):
        cdef float64[::1, :] predicted_and_quality_scores = self.predicted_and_quality_scores
        cdef float64[::1] sums_of_gradients = self.sums_of_gradients
        cdef float64 sum_of_hessians = self.sum_of_hessians
        cdef intp num_labels = sums_of_gradients.shape[0]
        cdef float64 sum_of_gradients, score, score_halved, sum_of_hessians_uncovered
        cdef float64[::1] total_sums_of_gradients
        cdef intp[::1] label_indices
        cdef intp c, l

        if include_uncovered:
            sum_of_hessians_uncovered = self.total_sum_of_hessians - sum_of_hessians
            total_sums_of_gradients = self.total_sums_of_gradients
            label_indices = self.label_indices

        for c in range(num_labels):
            sum_of_gradients = sums_of_gradients[c]

            # Calculate score to be predicted by a rule that covers the examples that have been provided so far...
            score = divide(-sum_of_gradients, sum_of_hessians)
            predicted_and_quality_scores[0, c] = score

            # Calculate quality score for a rule that covers the examples that have been provided so far..
            score_halved = score / 2
            predicted_and_quality_scores[1, c] = (sum_of_gradients * score) + (score_halved * sum_of_hessians * score)

            if include_uncovered:
                # Calculate score to be predicted by a rule that covers the examples that have not been provided yet...
                l = __get_label_index(c, label_indices)
                score = divide(-(total_sums_of_gradients[l] - sum_of_gradients), sum_of_hessians_uncovered)
                predicted_and_quality_scores[2, c] = score

                # Calculate quality score for a rule that covers the examples that have not been provided yet...
                score_halved = score / 2
                predicted_and_quality_scores[3, c] = ((total_sums_of_gradients[l] - sum_of_gradients) * score) + (score_halved * sum_of_hessians_uncovered * score)

        return predicted_and_quality_scores

    cdef float64[::1] calculate_predicted_scores(self):
        cdef float64[::1] sums_of_gradients = self.sums_of_gradients
        cdef float64 sum_of_hessians = self.sum_of_hessians
        cdef intp num_labels = sums_of_gradients.shape[0]
        cdef float64[::1] predicted_scores = cvarray(shape=(num_labels,), itemsize=sizeof(float64), format='d',
                                                     mode='c')
        cdef float64 sum_of_gradients, score

        for c in range(num_labels):
            sum_of_gradients = sums_of_gradients[c]

            # Calculate score to be predicted by a rule that covers the examples that have been provided so far...
            score = -sum_of_gradients / sum_of_hessians
            predicted_scores[c] = score

        return predicted_scores

    cdef apply_predictions(self, intp[::1] covered_example_indices, intp[::1] label_indices,
                           float64[::1] predicted_scores):
        cdef intp num_labels = label_indices.shape[0]
        cdef float64[::1, :] gradients = self.gradients
        cdef float64[::1] total_sums_of_gradients = self.total_sums_of_gradients
        cdef float64 total_sum_of_gradients, predicted_score, gradient
        cdef intp c, l, i

        # Only the labels that are predicted by the new rule must be considered...
        for c in range(num_labels):
            l = label_indices[c]
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
        cdef float64 denominator = num_cols + 1
        cdef float64[::1, :] gradients = cvarray(shape=(num_rows, num_cols), itemsize=sizeof(float64), format='d',
                                                 mode='fortran')
        cdef float64[::1, :] hessians = cvarray(shape=(num_rows, num_cols), itemsize=sizeof(float64), format='d',
                                                mode='fortran')
        cdef float64[::1] scores = cvarray(shape=(num_cols,), itemsize=sizeof(float64), format='d', mode='c')
        cdef expected_scores = cvarray(shape=(num_cols,), itemsize=sizeof(float64), format='d', mode='c')
        cdef exponentials = cvarray(shape=(num_cols,), itemsize=sizeof(float64), format='d', mode='c')
        cdef float64 sum_of_gradients, sum_of_hessians, expected_score, exponential, sum_of_exponentials
        cdef intp r, c

        for c in range(num_cols):
            # Column-wise sum up gradients and hessians for the current label (this is possible, because the predicted
            # scores for each example and label are 0 initially)...
            sum_of_gradients = 0
            sum_of_hessians = 0

            for r in range(num_rows):
                expected_score = __convert_label_into_score(y[r, c])
                sum_of_gradients -= (expected_score / denominator)
                sum_of_hessians += ((pow(expected_score, 2) * num_cols) / denominator)

            # Calculate optimal score to be predicted by the default rule for the current label...
            scores[c] = -sum_of_gradients / sum_of_hessians

        # Traverse each row and column again to calculate the updated gradients and hessians (unlike before, they must
        # be calculated row-wise, because the loss function is applied example-wise)...
        for r in range(num_rows):
            sum_of_exponentials = 1

            # Iterate the labels of the current example once to initialize temporary variables that are shared among the
            # upcoming calculations to avoid redundant calculations...
            for c in range(num_cols):
                expected_score = __convert_label_into_score(y[r, c])
                expected_scores[c] = expected_score
                exponential = exp(-expected_score * scores[c])
                exponentials[c] = exponential
                sum_of_exponentials += exponential

            denominator = pow(sum_of_exponentials, 2)

            # Calculate updated gradients and hessians for each label of the current example...
            for c in range(num_cols):
                expected_score = expected_scores[c]
                exponential = exponentials[c]
                gradients[r, c] = (-expected_score * exponential) / sum_of_exponentials
                hessians[r, c] = (pow(expected_score, 2) * exponential * (sum_of_exponentials - exponential)) / denominator

        # Cache the matrix of gradients and the matrix of hessians...
        self.gradients = gradients
        self.hessians = hessians

        # TODO Cache the total sums of gradients and hessians for each label (if we can make use of it)...

        return scores

    cdef begin_instance_sub_sampling(self):
        # TODO
        pass

    cdef update_sub_sample(self, intp example_index):
        # TODO
        pass

    cdef begin_search(self, intp[::1] label_indices):
        # TODO
        pass

    cdef update_search(self, intp example_index, uint32 weight):
        # TODO
        pass

    cdef float64[::1, :] calculate_predicted_and_quality_scores(self, bint include_uncovered):
        # TODO
        pass

    cdef float64[::1] calculate_predicted_scores(self):
        # TODO
        pass

    cdef apply_predictions(self, intp[::1] covered_example_indices, intp[::1] label_indices,
                           float64[::1] predicted_scores):
        # TODO
        pass