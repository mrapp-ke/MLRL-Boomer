from boomer.algorithm._model cimport intp, float32


# A struct that represents a condition of a rule. It consists of the index of the feature that is used by the condition,
# whether it uses the <= (leq=1) or > (leq=0) operator, as well as a threshold.
cdef struct s_condition:
    intp feature_index
    bint leq
    float32 threshold


cdef inline s_condition make_condition(intp feature_index, bint leq, float32 threshold):
    """
    Creates and returns a new condition.

    :param feature_index:   The index of the feature that is used by the condition
    :param leq:             Whether the <= operator, or the > operator is used by the condition
    :param threshold:       The threshold that is used by the condition
    """
    cdef s_condition condition
    condition.feature_index = feature_index
    condition.leq = leq
    condition.threshold = threshold
    return condition


cdef inline bint test_condition(float32 threshold, bint leq, float32 feature_value):
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
