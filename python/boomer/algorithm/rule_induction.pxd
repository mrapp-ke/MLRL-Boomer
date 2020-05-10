# distutils: language=c++
from boomer.algorithm._arrays cimport intp, uint8, float32
from boomer.algorithm.rules cimport Rule
from boomer.algorithm.losses cimport Loss
from boomer.algorithm.sub_sampling cimport InstanceSubSampling, FeatureSubSampling, LabelSubSampling
from boomer.algorithm.pruning cimport Pruning
from boomer.algorithm.shrinkage cimport Shrinkage
from boomer.algorithm.head_refinement cimport HeadRefinement

from libcpp.unordered_map cimport unordered_map as map


"""
A struct that stores a value of type float32 and a corresponding index that refers to the (original) position of the
value in an array.
"""
cdef struct IndexedElement:
    intp index
    float32 value


"""
A struct that contains a pointer to a C-array of type intp, representing the indices of the training examples that are
covered by a rule. The attribute `num_elements` specifies how many elements the array contains. The attribute
`num_conditions` specifies how many conditions the rule contained when the struct was updated for the last time. It may
be used to check if the array is still valid or must be updated.
"""
cdef struct IndexArray:
    intp* data
    intp num_elements
    intp num_conditions


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

    cdef Rule induce_default_rule(self, uint8[::1, :] y, Loss loss)

    cdef Rule induce_rule(self, intp[::1] nominal_attribute_indices, float32[::1, :] x, uint8[::1, :] y,
                          HeadRefinement head_refinement, Loss loss, LabelSubSampling label_sub_sampling,
                          InstanceSubSampling instance_sub_sampling, FeatureSubSampling feature_sub_sampling,
                          Pruning pruning, Shrinkage shrinkage, intp random_state)


cdef class ExactGreedyRuleInduction(RuleInduction):

    # Attributes:

    cdef map[intp, intp*]* sorted_indices_map_global

    # Functions:

    cdef Rule induce_default_rule(self, uint8[::1, :] y, Loss loss)

    cdef Rule induce_rule(self, intp[::1] nominal_attribute_indices, float32[::1, :] x, uint8[::1, :] y,
                          HeadRefinement head_refinement, Loss loss, LabelSubSampling label_sub_sampling,
                          InstanceSubSampling instance_sub_sampling, FeatureSubSampling feature_sub_sampling,
                          Pruning pruning, Shrinkage shrinkage, intp random_state)


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
