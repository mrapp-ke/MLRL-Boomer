# distutils: language=c++
from boomer.algorithm._arrays cimport intp, uint8, float32
from boomer.algorithm.rules cimport Rule
from boomer.algorithm.losses cimport Loss
from boomer.algorithm.sub_sampling cimport InstanceSubSampling, FeatureSubSampling, LabelSubSampling
from boomer.algorithm.pruning cimport Pruning
from boomer.algorithm.shrinkage cimport Shrinkage
from boomer.algorithm.head_refinement cimport HeadRefinement


"""
An enum that specifies all possible types of operators used by a condition of a rule.
"""
cdef enum Comparator:
    LEQ = 0
    GR = 1
    EQ = 2
    NEQ = 3


"""
A struct that represents a condition of a rule. It consists of the index of the feature the condition corresponds to,
the type of the operator that is used by the condition, as well as a threshold.
"""
cdef struct Condition:
    intp feature_index
    Comparator comparator
    float32 threshold


cdef class RuleInduction:

    # Functions:

    cpdef Rule induce_default_rule(self, uint8[::1, :] y, Loss loss)

    cpdef Rule induce_rule(self, intp[::1] nominal_attribute_indices, float32[::1, :] x, intp[::1, :] x_sorted_indices,
                           uint8[::1, :] y, HeadRefinement head_refinement, Loss loss,
                           LabelSubSampling label_sub_sampling, InstanceSubSampling instance_sub_sampling,
                           FeatureSubSampling feature_sub_sampling, Pruning pruning, Shrinkage shrinkage,
                           random_state: int)


cdef class ExactGreedyRuleInduction(RuleInduction):

    # Functions:

    cpdef Rule induce_default_rule(self, uint8[::1, :] y, Loss loss)

    cpdef Rule induce_rule(self, intp[::1] nominal_attribute_indices, float32[::1, :] x, intp[::1, :] x_sorted_indices,
                           uint8[::1, :] y, HeadRefinement head_refinement, Loss loss,
                           LabelSubSampling label_sub_sampling, InstanceSubSampling instance_sub_sampling,
                           FeatureSubSampling feature_sub_sampling, Pruning pruning, Shrinkage shrinkage,
                           random_state: int)


cdef inline bint test_condition(float32 threshold, Comparator comparator, float32 feature_value):
    """
    Returns whether a given feature value satisfies a certain condition.

    :param threshold:       The threshold of the condition
    :param comparator:      The operator that is used by the condition
    :param feature_value:   The feature value
    :return:                1, if the feature value satisfies the condition, 0 otherwise
    """
    if comparator == Comparator.LEQ:
        return feature_value <= threshold
    elif comparator == Comparator.GR:
        return feature_value > threshold
    elif comparator == Comparator.EQ:
        return feature_value == threshold
    else:
        return feature_value != threshold
