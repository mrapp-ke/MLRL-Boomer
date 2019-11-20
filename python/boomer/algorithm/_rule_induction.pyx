# distutils: language=c++
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=False
from cython.view cimport array as cvarray
from boomer.algorithm._model cimport intp, uint8, uint32, float32, float64
from boomer.algorithm._model cimport Rule, FullHead, EmptyBody, ConjunctiveBody, PartialHead
from boomer.algorithm._head_refinement cimport HeadCandidate, HeadRefinement
from boomer.algorithm._losses cimport Loss
from boomer.algorithm._sub_sampling cimport InstanceSubSampling, FeatureSubSampling

from libcpp.unordered_map cimport unordered_map as map
from cython.operator cimport dereference, postincrement

import numpy as np
from boomer.algorithm.model import DTYPE_INTP, DTYPE_FLOAT32


cpdef Rule induce_default_rule(uint8[::1, :] y, Loss loss):
    """
    Induces the default rule that minimizes a certain loss function with respect to the expected confidence scores
    according to the ground truth.

    :param y:       An array of dtype float, shape `(num_examples, num_labels)`, representing the labels of the training
                    examples
    :param loss:    The loss function to be minimized
    :return:        The default rule that has been induced
    """
    cdef float64[::1] scores = loss.calculate_default_scores(y)
    cdef FullHead head = FullHead(scores)
    cdef EmptyBody body = EmptyBody()
    cdef Rule rule = Rule(body, head)
    return rule


cpdef Rule induce_rule(float32[::1, :] x, intp[::1, :] x_sorted_indices, HeadRefinement head_refinement, Loss loss,
                       InstanceSubSampling instance_sub_sampling, FeatureSubSampling feature_sub_sampling,
                       random_state: int):
    """
    Induces a single- or multi-label classification rule that minimizes a certain loss function with respect to the
    expected and currently predicted confidence scores.

    :param x:                       An array of dtype float, shape `(num_examples, num_features)`, representing the
                                    features of the training examples
    :param x_sorted_indices:        An array of dtype int, shape `(num_examples, num_features)`, representing the
                                    indices of the examples when sorting column-wise
    :param head_refinement:         The strategy that is used to find the heads of rules
    :param loss:                    The loss function to be minimized
    :param instance_sub_sampling:   The strategy that should be used to sub-sample the training examples or None, if no
                                    instance sub-sampling should be used
    :param feature_sub_sampling:    The strategy that should be used to sub-sample the available features or None, if no
                                    feature sub-sampling should be used
    :param random_state:            The seed to be used by RNGs
    :return:                        The rule that has been induced
    """
    # Sub-sample examples, if necessary...
    cdef uint32[::1] weights

    if instance_sub_sampling is None:
        weights = None
    else:
        weights = instance_sub_sampling.sub_sample(x, loss, random_state)

    # The head of the induced rule
    cdef HeadCandidate head = None
    # A map containing the feature indices of the rule's conditions that use the <= operator as keys and their
    # thresholds as values
    cdef map[intp, float32] leq_conditions
    # A map containing the feature indices of the rule's conditions that use the > operator as keys and their thresholds
    # as values
    cdef map[intp, float32] gr_conditions

    # Variables used to update the seed used by RNGs, depending on the refinement iteration (starting at 1)
    cdef int current_random_state = random_state
    cdef int num_refinements = 1

    # Variables for representing the best refinement
    cdef HeadCandidate best_head
    cdef bint best_condition_leq
    cdef intp best_condition_r, best_condition_index
    cdef float32 best_condition_threshold

    # Variables for specifying the examples and labels that should be used for finding the best refinement
    cdef intp[::1] label_indices = None
    cdef intp[::1, :] sorted_indices = x_sorted_indices
    cdef intp num_examples

    # Variables for specifying the features used for finding the best refinement
    cdef intp[::1] feature_indices
    cdef intp num_features

    # Temporary variables
    cdef HeadCandidate current_head
    cdef float32 previous_threshold, current_threshold
    cdef float64[::1, :] predicted_and_quality_scores
    cdef uint32 weight
    cdef intp c, f, r, i, offset

    # Search for the best refinement until no improvement in terms of the rule's quality score is possible anymore...
    while True:
        num_examples = sorted_indices.shape[0]
        best_head = None

        # Sub-sample features, if necessary...
        if feature_sub_sampling is None:
            feature_indices = None
            num_features = x.shape[1]
        else:
            feature_indices = feature_sub_sampling.sub_sample(x, current_random_state)
            num_features = feature_indices.shape[0]

        # Search for the best condition among all available features to be added to the current rule...
        for c in range(num_features):
            # For each feature, the examples are traversed in increasing order of their respective feature values and
            # the loss function is updated accordingly.
            f = __get_feature_index(c, feature_indices)

            # Reset the loss function when processing a new feature...
            loss.begin_search(label_indices)

            # Find first example with weight > 0...
            for r in range(0, num_examples):
                i = sorted_indices[r, f]
                weight = __get_weight(i, weights)

                if weight > 0:
                    # Tell the loss function that the example will be covered by upcoming conditions...
                    loss.update_search(i, weight)
                    previous_threshold = x[i, f]
                    break

            # Traverse remaining instances...
            offset = r + 1

            for r in range(offset, num_examples):
                i = sorted_indices[r, f]
                weight = __get_weight(i, weights)

                if weight > 0:
                    current_threshold = x[i, f]

                    # Split points between examples with the same feature value must not be considered...
                    if previous_threshold != current_threshold:
                        # Calculate optimal scores to be predicted by the current refinement, as well as the
                        # corresponding quality scores
                        predicted_and_quality_scores = loss.calculate_predicted_and_quality_scores()

                        # Evaluate potential condition using <= operator...
                        current_head = head_refinement.find_head(head, best_head, loss, predicted_and_quality_scores, 0)

                        if current_head is not None:
                            best_head = current_head
                            best_condition_leq = 1
                            best_condition_r = r
                            best_condition_index = f
                            best_condition_threshold = __calculate_threshold(previous_threshold, current_threshold)

                        # Evaluate potential condition using > operator...
                        current_head = head_refinement.find_head(head, best_head, loss, predicted_and_quality_scores, 2)

                        if current_head is not None:
                            best_head = current_head
                            best_condition_leq = 0
                            best_condition_r = r
                            best_condition_index = f
                            best_condition_threshold = __calculate_threshold(previous_threshold, current_threshold)

                    # Tell the loss function that the example will be covered by upcoming conditions...
                    loss.update_search(i, weight)
                    previous_threshold = current_threshold

        if best_head is None:
            break
        else:
            # Apply refinement to the current rule...
            head = best_head

            if best_condition_leq:
                leq_conditions[best_condition_index] = best_condition_threshold
            else:
                gr_conditions[best_condition_index] = best_condition_threshold

            if num_examples <= 1:
                # Abort refinement process if rule covers a single instance...
                break
            else:
                # Otherwise, prepare next refinement iteration by updating the examples and labels that should be used
                # for finding the next refinement...
                label_indices = head.label_indices
                sorted_indices = __filter_sorted_indices(x, sorted_indices, best_condition_r, best_condition_index,
                                                         best_condition_leq, best_condition_threshold)
                num_examples = sorted_indices.shape[0]

                # Tell the loss function that a new rule has been induced...
                loss.apply_predictions(sorted_indices[:, 0], label_indices, head.predicted_scores)

                # Alter seed to be used by RNGs for the next refinement...
                num_refinements += 1
                current_random_state = random_state * num_refinements

    # Build and return the induced rule...
    return __build_rule(head, leq_conditions, gr_conditions)

cdef Rule __build_rule(HeadCandidate head, map[intp, float32] leq_conditions, map[intp, float32] gr_conditions):
    """
    Builds and returns a rule.

    :param head:            A 'HeadCandidate' representing the head of the rule
    :param leq_conditions:  A map that contains the feature indices of the rule's conditions that use the <= operator as
                            keys and their thresholds as values
    :param gr_conditions:   A map that contains the feature indices of the rule's conditions that use the > operator as
                            keys and their thresholds as values
    :return:                The rule that has been built
    """
    cdef intp num_leq_conditions = leq_conditions.size()
    cdef intp[::1] leq_feature_indices = np.empty((num_leq_conditions,), DTYPE_INTP, 'C')
    cdef float32[::1] leq_thresholds = np.empty((num_leq_conditions,), DTYPE_FLOAT32, 'C')
    cdef intp num_gr_conditions = gr_conditions.size()
    cdef intp[::1] gr_feature_indices = np.empty((num_gr_conditions,), DTYPE_INTP, 'C')
    cdef float32[::1] gr_thresholds = np.empty((num_gr_conditions,), DTYPE_FLOAT32, 'C')

    cdef map[intp, float32].iterator iterator = leq_conditions.begin()
    cdef intp index
    cdef float32 threshold
    cdef intp i = 0

    while iterator != leq_conditions.end():
        index = dereference(iterator).first
        threshold = dereference(iterator).second
        leq_feature_indices[i] = index
        leq_thresholds[i] = threshold
        postincrement(iterator)
        i += 1

    iterator = gr_conditions.begin()
    i = 0

    while iterator != gr_conditions.end():
        index = dereference(iterator).first
        threshold = dereference(iterator).second
        gr_feature_indices[i] = index
        gr_thresholds[i] = threshold
        postincrement(iterator)
        i += 1

    cdef ConjunctiveBody rule_body = ConjunctiveBody(leq_feature_indices, leq_thresholds, gr_feature_indices,
                                                     gr_thresholds)
    cdef PartialHead rule_head = PartialHead(head.label_indices, head.predicted_scores)
    cdef Rule rule = Rule(rule_body, rule_head)
    return rule

cdef intp[::1, :] __filter_sorted_indices(float32[::1, :] x, intp[::1, :] sorted_indices, intp condition_r,
                                          intp condition_index, bint condition_leq, float32 condition_threshold):
    """
    Filters the matrix of example indices after a new condition has been added to a previous rule, such that the
    filtered matrix does only contain the indices of examples that are covered by the new rule.

    :param x:                   An array of dtype float, shape `(num_examples, num_features)`, representing the features
                                of the training examples
    :param x_sorted_indices:    An array of dtype int, shape `(num_examples, num_features)`, representing the indices of
                                the examples that are covered by the previous rule when sorting column-wise
    :param condition_r:         The index of the example from which the threshold of the condition that has been added
                                to the previous rule has been chosen
    :param condition_index:     The the feature index of the condition that has been added to the previous rule
    :param condition_leq:       1, if the condition that has been added to the previous rule uses the <= operator, 0, if
                                the condition uses the > operator
    :param condition_threshold: The threshold of the condition that has been added to the previous rule
    :return:                    An array of dtype int, shape `(num_covered_examples, num_features)`, representing the
                                indices of the examples that are covered by the new rule when sorting column-wise
    """
    cdef intp num_features = x.shape[1]
    cdef intp num_examples = sorted_indices.shape[0]
    cdef intp num_covered

    if condition_leq:
        num_covered = condition_r
    else:
        num_covered = num_examples - condition_r

    cdef intp[::1, :] filtered_sorted_indices = cvarray(shape=(num_covered, num_features), itemsize=sizeof(intp),
                                                        format='l', mode='fortran')
    cdef float32 feature_value
    cdef intp c, r, i, index

    for c in range(num_features):
        i = 0

        if c == condition_index:
            # For the feature used by the new condition we know the indices of the covered examples...
            if condition_leq:
                offset = 0
            else:
                offset = condition_r

            for r in range(offset, offset + num_covered):
                index = sorted_indices[r, c]
                filtered_sorted_indices[i, c] = index
                i += 1
        else:
            # For the other features we need to filter out the indices that correspond to examples that do not satisfy
            # the new condition...
            for r in range(num_examples):
                index = sorted_indices[r, c]
                feature_value = x[index, condition_index]

                if __test_condition(condition_threshold, condition_leq, feature_value):
                    filtered_sorted_indices[i, c] = index
                    i += 1

                    if i >= num_covered:
                        break

    return filtered_sorted_indices


cdef inline intp __get_feature_index(intp i, intp[::1] feature_indices):
    """
    Retrieves and returns the index of the i-th feature from an array of feature indices, if such an array is available.
    Otherwise i is returned.

    :param i:               The position of the feature whose index should be retrieved
    :param label_indices:   An array of the dtype int, shape `(num_features)`, representing the indices of features
    :return:                A scalar of dtype int, representing the index of the i-th feature
    """
    if feature_indices is None:
        return i
    else:
        return feature_indices[i]


cdef inline uint32 __get_weight(intp example_index, uint32[::1] weights):
    """
    Retrieves and returns the weight of the example at a specific index from an array of weights, if such an array is
    available.

    :param example_index:   The index of the example whose weight should be retrieved
    :param weights:         An array of dtype int, shape `(num_examples)`, representing the weights of examples
    :return:                A scalar of dtype int, representing the weight of the example at the given index
    """
    if weights is None:
        return 1
    else:
        return weights[example_index]


cdef inline float32 __calculate_threshold(float32 previous_threshold, float32 current_threshold):
    """
    Calculates and returns the threshold to be used by a rule's condition, given the largest feature value of the
    covered examples and the smallest feature value of the uncovered examples.

    :param previous_threshold:  A scalar of dtype float, representing the largest feature value of the covered examples
    :param current_threshold:   A scalar of dtype float, representing the smallest feature value of the uncovered
                                examples
    :return:                    A scalar of dtype float, representing the calculated threshold
    """
    return previous_threshold + ((current_threshold - previous_threshold) / 2)


cdef inline bint __test_condition(float32 threshold, bint leq, float32 feature_value):
    """
    Returns whether a given feature value satisfies a certain condition.

    :param threshold:       The threshold of the condition
    :param leq:             1, if the condition uses the <= operator, 0, if it uses the > operator
    :param feature_value:   The feature value
    :return:                1, if the feature value satisfies the condition, 0 otherwise
    """
    if leq:
        return feature_value <= threshold
    else:
        return feature_value > threshold
