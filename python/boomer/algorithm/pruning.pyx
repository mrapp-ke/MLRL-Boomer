# distutils: language=c++

"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides classes that implement strategies for pruning classification rules.
"""
from boomer.algorithm._arrays cimport float32, float64
from boomer.algorithm.losses cimport Prediction
from boomer.algorithm.rule_induction cimport Comparator

from cython.operator cimport dereference, postincrement


cdef class Pruning:
    """
    A base class for all classes that implement a strategy for pruning classification rules based on a "prune set",
    i.e., based on the examples that are not contained in the sub-sample that has been used to grow the rule (referred
    to as the "grow set").
    """

    cdef pair[uint32[::1], uint32] prune(self, map[intp, IndexedArray*]* sorted_feature_values_map,
                                         list[Condition] conditions, uint32[::1] covered_examples_mask,
                                         uint32 covered_examples_target, uint32[::1] weights, intp[::1] label_indices,
                                         Loss loss, HeadRefinement head_refinement):
        """
        Prunes the conditions of an existing rule by modifying a given list of conditions in-place. The rule is pruned
        by removing individual conditions in a way that improves over its original quality score as measured on the
        "prune set".

        :param sorted_feature_values_map:   A pointer to a map that maps feature indices to structs of type
                                            `IndexedArray`, storing the indices of all training examples, as well as
                                            their values for the respective feature, sorted in ascending order by the
                                            feature values
        :param conditions:                  A list that contains the conditions of the existing rule
        :param covered_examples_mask:       An array of dtype uint, shape `(num_examples)` that is used to keep track of
                                            the indices of the examples that are covered by the existing rule
        :param covered_examples_target:     The value that is used to mark those elements in `covered_examples_mask`
                                            that are covered by the existing rule
        :param weights:                     An array of dtype int, shape `(num_examples)`, representing the weights of
                                            all training examples, regardless of whether they are included in the prune
                                            set or grow set
        :param label_indices:               An array of dtype int, shape `(num_predicted_labels)`, representing the
                                            indices of the labels for which the existing rule predicts or None, if the
                                            rule predicts for all labels
        :param loss:                        The loss function to be minimized
        :param head_refinement:             The strategy that is used to find the heads of rules
        """
        pass


cdef class IREP(Pruning):
    """
    Implements incremental reduced error pruning (IREP) for pruning classification rules based on a "prune set".

    Given a rule with n conditions, IREP allows to remove up to n - 1 trailing conditions, depending on which of the
    pruning candidates improves the most over the overall quality score of the original rule (calculated on the prune
    set).
    """

    cdef pair[uint32[::1], uint32] prune(self, map[intp, IndexedArray*]* sorted_feature_values_map,
                                         list[Condition] conditions, uint32[::1] covered_examples_mask,
                                         uint32 covered_examples_target, uint32[::1] weights, intp[::1] label_indices,
                                         Loss loss, HeadRefinement head_refinement):
        # The total number of training examples
        cdef intp num_examples = covered_examples_mask.shape[0]
        # The number of conditions of the existing rule
        cdef intp num_conditions = conditions.size()
        # Temporary variables
        cdef Prediction prediction
        cdef Condition condition
        cdef Comparator comparator
        cdef float32 threshold
        cdef uint32 weight
        cdef intp feature_index, i, n

        # Reset the loss function...
        loss.begin_instance_sub_sampling()
        loss.begin_search(label_indices)

        # Tell the loss function about all examples in the prune set that are covered by the existing rule...
        for i in range(num_examples):
            weight = weights[i]

            if weight == 0:
                loss.update_sub_sample(i, 1, False)

                if covered_examples_mask[i] == covered_examples_target:
                    loss.update_search(i, 1)

        # Determine the optimal prediction of the existing rule, as well as the corresponding quality score, based on
        # the prune set...
        prediction = head_refinement.evaluate_predictions(loss, False, False)
        cdef float64 original_quality_score = prediction.overall_quality_score

        # We process the existing rule's conditions (except for the last one) in the order they have been learned. At
        # each iteration, we calculate the quality score of a rule that only contains the conditions processed so far
        # and keep track of the best rule...
        cdef list[Condition].iterator iterator = conditions.begin()

        for n in range(num_conditions - 1):
            # Obtain properties of the current condition...
            condition = dereference(iterator)
            feature_index = condition.feature_index
            threshold = condition.threshold
            comparator = condition.comparator

            postincrement(iterator)

        # TODO Implement pruning

        cdef pair[uint32[::1], uint32] result
        result.first = covered_examples_mask
        result.second = covered_examples_target
        return result
