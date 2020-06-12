# distutils: language=c++

"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides classes that implement strategies for pruning classification rules.
"""
from boomer.algorithm._arrays cimport array_intp
from boomer.algorithm.rule_induction cimport Comparator
from boomer.algorithm.losses cimport Prediction

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
        # TODO Implement
        cdef pair[uint32[::1], uint32] result
        result.first = covered_examples_mask
        result.second = covered_examples_target
        return result
