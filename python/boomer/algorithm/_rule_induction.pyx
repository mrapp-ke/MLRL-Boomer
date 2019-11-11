from boomer.algorithm._model cimport intp, uint8, uint32, float32, float64, Rule, FullHead, EmptyBody
from boomer.algorithm._head_refinement cimport HeadCandidate, HeadRefinement
from boomer.algorithm._losses cimport Loss
from boomer.algorithm._sub_sampling cimport InstanceSubSampling, FeatureSubSampling

from libcpp.unordered_map cimport unordered_map as map


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
    cdef int current_random_state = random_state
    cdef uint32[::1] weights

    # Sub-sample examples, if necessary...
    if instance_sub_sampling is None:
        weights = None
    else:
        weights = instance_sub_sampling.sub_sample(x, loss, current_random_state)

    # The head of the induced rule
    cdef HeadCandidate best_head = None
    # A map containing the feature indices of the rule's conditions that use the <= operator as keys and their
    # thresholds as values
    cdef map[intp, float32] leq_conditions
    # A map containing the feature indices of the rule's conditions that use the > operator as keys and their thresholds
    # as values
    cdef map[intp, float32] gr_conditions

    cdef int num_refinements = 1
    cdef intp[::1] label_indices = None
    cdef intp num_examples = x.shape[0]
    cdef intp num_features
    cdef intp[::1] feature_indices
    cdef float32 previous_threshold, current_threshold
    cdef HeadCandidate head_candidate
    cdef intp c, f, r, i
    cdef uint32 weight

    while True:
        # Sub-sample features, if necessary...
        if feature_sub_sampling is None:
            feature_indices = None
            num_features = x.shape[1]
        else:
            feature_indices = feature_sub_sampling.sub_sample(x, current_random_state)
            num_features = feature_indices.shape[0]

        for c in range(num_features):
            loss.begin_search(label_indices)
            f = __get_feature_index(c, feature_indices)

            # Find first example with weight > 0...
            weight = 0

            for r in range(0, num_examples - 1):
                i = x_sorted_indices[r, f]
                weight = __get_weight(i, weights)

                if weight > 0:
                    loss.update_search(i, weight)
                    previous_threshold = x[i, f]
                    break

            # Traverse remaining instances (except for the last one)...
            for r in range(r + 1, num_examples - 1):
                i = x_sorted_indices[r, f]
                weight = __get_weight(i, weights)

                if weight > 0:
                    loss.update_search(i, weight)
                    current_threshold = x[i, f]

                    if previous_threshold != current_threshold:
                        # Evaluate condition using <= operator...
                        head_candidate = head_refinement.find_head(best_head, loss, 1)

                        if head_candidate is not None:
                            best_head = head_candidate

                        # Evaluate condition using > operator...
                        head_candidate = head_refinement.find_head(best_head, loss, 0)

                        if head_candidate is not None:
                            best_head = head_candidate

            previous_threshold = current_threshold

        num_refinements += 1
        current_random_state *= num_refinements
        break # TODO Use proper stopping criterion

    return None

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
