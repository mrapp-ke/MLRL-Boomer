from boomer.algorithm._model cimport float32


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
