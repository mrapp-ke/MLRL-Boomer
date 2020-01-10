# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides commonly used utility functions and structs.
"""
from boomer.algorithm._arrays cimport intp, float32, float64


"""
A struct that represents a condition of a rule. It consists of the index of the feature that is used by the condition,
whether it uses the <= (leq=1) or > (leq=0) operator, as well as a threshold.
"""
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


cdef inline float64 divide_or_zero(float64 a, float64 b):
    """
    Divides a floating point number by another one. The division by zero evaluates to 0 per definition.

    :param a: The number to be divided
    :param b: The divisor
    :return:  The result of a / b or 0, if b = 0
    """
    if b != 0:
        return a / b
    else:
        return 0


cdef inline intp get_label_index(intp i, intp[::1] label_indices):
    """
    Retrieves and returns the index of the i-th label from an array of label indices, if such an array is available.
    Otherwise i is returned.

    :param i:               The position of the label whose index should be retrieved
    :param label_indices:   An array of the dtype int, shape `(num_labels)`, representing the indices of labels
    :return:                A scalar of dtype int, representing the index of the i-th label
    """
    if label_indices is None:
        return i
    else:
        return label_indices[i]
