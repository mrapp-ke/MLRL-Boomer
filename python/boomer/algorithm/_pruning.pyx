# distutils: language=c++
# cython: boundscheck=False
# cython: wraparound=False
from boomer.algorithm._model cimport float64


cdef class Pruning:
    """
    A base class for all classes that implement a strategy for pruning classification rules based on a "prune set",
    i.e., based on the examples that are not contained in the sub-sample that has been used to grow the rule (referred
    to as the "grow set").
    """

    cdef begin_pruning(self, uint32[::1] weights, Loss loss, intp[::1] covered_example_indices, intp[::1] label_indices,
                       float64[::1] predicted_scores):
        """
        Calculates the quality score of an existing rule, based on the examples that are contained in the prune set,
        i.e., based on all examples whose weight is 0.

        This function must be called prior to calling any other function provided by this class. It calculates and
        caches the original quality score of the existing rule before it is pruned. When invoking the function `prune`
        afterwards, the rule is pruned by removing individual conditions in a way that improves over the original
        quality score, if possible.

        :param weights:                 An array of dtype int, shape `(num_examples)`, representing the weights of all
                                        training examples, regardless of whether they are included in the prune set or
                                        grow set
        :param loss:                    The loss function to be minimized
        :param covered_example_indices: An array of dtype int, shape `(num_covered_examples)`, representing the indices
                                        of all training examples that are covered by the rule, regardless of whether
                                        they are included in the prune set or grow set
        :param label_indices:           An array of dtype int, shape `(num_predicted_labels)`, representing the indices
                                        of the labels for which the rule predicts
        :param predicted_scores:        An array of dtype int, shape `(num_predicted_labels)`, representing the scores
                                        that are predicted by the rule for the labels in `label_indices`
        """
        pass

    cdef prune(self, float32[::1, :] x, intp[::1, :] x_sorted_indices, uint32[::1] weights, Loss loss,
               list[s_condition] conditions, intp[::1] covered_indices):
        """
        Prunes a rule on the examples that are not contained in the sub-sample that has been used to grow the rule,
        i.e., for which the weight is 0.

        :param x:                   An array of dtype float, shape `(num_examples, num_features)`, representing the
                                    features of the training examples
        :param x_sorted_indices:    An array of dtype int, shape `(num_examples, num_features)`, representing the
                                    indices of the training examples when sorting column-wise
        :param y:                   An array of dtype float, shape `(num_examples, num_labels)`, representing the labels
                                    of the training examples
        :param weights:             An array of dtype uint, shape `(num_examples)`, representing the weights of the
                                    given training examples, i.e., how many times each of the examples is contained in
                                    the sample
        :param loss:                The loss function to be minimized
        :param conditions:          A list that contains the rule's conditions
        :param covered_indices:     An array of dtype int, shape `(num_covered_examples)`, representing the indices of
                                    the examples that are covered by the rule
        """
        pass


cdef class IREP(Pruning):
    """
    Implements incremental reduced error pruning (IREP) for pruning classification rules based on a "prune set".
    """

    cdef begin_pruning(self, uint32[::1] weights, Loss loss, intp[::1] covered_example_indices, intp[::1] label_indices,
                       float64[::1] predicted_scores):
        cdef uint32 weight
        cdef intp i

        # Reset the loss function...
        loss.begin_search(label_indices)

        # Tell the loss function about all examples in the prune set that are covered by the given rule...
        for i in covered_example_indices:
            weight = weights[i]

            if weight == 0:
                loss.update_search(i, 1)

        # Calculate the overall quality score of the given rule based on the prune set...
        cdef float64 original_quality_score = loss.calculate_quality_score(predicted_scores)
        self.original_quality_score = original_quality_score

    cdef prune(self, float32[::1, :] x, intp[::1, :] x_sorted_indices, uint32[::1] weights, Loss loss,
               list[s_condition] conditions, intp[::1] covered_indices):
        # TODO
        pass
