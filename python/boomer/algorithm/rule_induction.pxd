# distutils: language=c++
from boomer.algorithm._arrays cimport intp, uint8, float32
from boomer.algorithm._random cimport RNG
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
cdef struct IndexedValue:
    intp index
    float32 value


"""
A struct that contains a pointer to a C-array of type `IndexedValue`. The attribute `num_elements` specifies how many
elements the array contains.
"""
cdef struct IndexedArray:
    IndexedValue* data
    intp num_elements


"""
A struct that contains a pointer to a struct of type `IndexedArray`, representing the indices and feature values of the
training examples that are covered by a rule. The attribute `num_conditions` specifies how many conditions the rule
contained when the array was updated for the last time. It may be used to check if the array is still valid or must be
updated.
"""
cdef struct IndexedArrayWrapper:
    IndexedArray* array
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

    cdef Rule induce_rule(self, intp[::1] nominal_attribute_indices, float32[::1] x_data, intp[::1] x_row_indices,
                          intp[::1] x_col_indices, intp num_examples, intp num_labels, HeadRefinement head_refinement,
                          Loss loss, LabelSubSampling label_sub_sampling, InstanceSubSampling instance_sub_sampling,
                          FeatureSubSampling feature_sub_sampling, Pruning pruning, Shrinkage shrinkage,
                          intp min_coverage, intp max_conditions, RNG rng)


cdef class ExactGreedyRuleInduction(RuleInduction):

    # Attributes:

    cdef map[intp, IndexedArray*]* cache_global

    # Functions:

    cdef Rule induce_default_rule(self, uint8[::1, :] y, Loss loss)

    cdef Rule induce_rule(self, intp[::1] nominal_attribute_indices, float32[::1] x_data, intp[::1] x_row_indices,
                          intp[::1] x_col_indices, intp num_examples, intp num_labels, HeadRefinement head_refinement,
                          Loss loss, LabelSubSampling label_sub_sampling, InstanceSubSampling instance_sub_sampling,
                          FeatureSubSampling feature_sub_sampling, Pruning pruning, Shrinkage shrinkage,
                          intp min_coverage, intp max_conditions, RNG rng)
