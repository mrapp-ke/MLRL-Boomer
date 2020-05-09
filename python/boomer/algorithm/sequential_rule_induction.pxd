from boomer.algorithm._arrays cimport uint8, intp, float32
from boomer.algorithm.rule_induction cimport RuleInduction
from boomer.algorithm.head_refinement cimport HeadRefinement
from boomer.algorithm.losses cimport Loss
from boomer.algorithm.pruning cimport Pruning
from boomer.algorithm.shrinkage cimport Shrinkage
from boomer.algorithm.sub_sampling cimport InstanceSubSampling, FeatureSubSampling, LabelSubSampling


cdef class SequentialRuleInduction:

    # Functions:

    cpdef object induce_rules(self, intp[::1] nominal_attribute_indices, float32[::1, :] x,
                              intp[::1, :] x_sorted_indices, uint8[::1, :] y, intp random_state)


cdef class RuleListInduction(SequentialRuleInduction):

    # Attributes:

    cdef bint default_rule_at_end

    cdef RuleInduction rule_induction

    cdef HeadRefinement head_refinement

    cdef Loss loss

    cdef list stopping_criteria

    cdef LabelSubSampling label_sub_sampling

    cdef InstanceSubSampling instance_sub_sampling

    cdef FeatureSubSampling feature_sub_sampling

    cdef Pruning pruning

    cdef Shrinkage shrinkage

    # Functions:

    cpdef object induce_rules(self, intp[::1] nominal_attribute_indices, float32[::1, :] x,
                              intp[::1, :] x_sorted_indices, uint8[::1, :] y, intp random_state)
