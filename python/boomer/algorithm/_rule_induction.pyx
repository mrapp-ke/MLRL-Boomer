from boomer.algorithm._model cimport intp, uint8, float32, float64, Rule, FullHead, EmptyBody
from boomer.algorithm._head_refinement cimport HeadRefinement
from boomer.algorithm._losses cimport Loss
from boomer.algorithm._sub_sampling cimport InstanceSubSampling, FeatureSubSampling

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
    cdef intp num_examples = x.shape[0]
    cdef intp num_features = x.shape[1]
    cdef float32 previous_threshold, current_threshold
    cdef intp r, c, i

    # TODO Sub-sample instances if necessary
    # TODO Sub-sample features if necessary
    for c in range(num_features):
        loss.begin_search(None)
        i = x_sorted_indices[0, c]
        loss.update_search(i, 1) # TODO: Use actual weight
        previous_threshold = x[i, c]

        for r in range(1, num_examples - 1):
            i = x_sorted_indices[r, c]
            loss.update_search(i, 1) # TODO Use actual weight
            current_threshold = x[i, c]

            #if previous_threshold != current_threshold:
                # LEQ
                # GR

            previous_threshold = current_threshold

    return None
