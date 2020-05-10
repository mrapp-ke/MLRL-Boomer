# distutils: language=c++
from boomer.algorithm._arrays cimport intp, uint8, float32
from boomer.algorithm.rules cimport Rule
from boomer.algorithm.losses cimport Loss
from boomer.algorithm.sub_sampling cimport InstanceSubSampling, FeatureSubSampling, LabelSubSampling
from boomer.algorithm.pruning cimport Pruning
from boomer.algorithm.shrinkage cimport Shrinkage
from boomer.algorithm.head_refinement cimport HeadRefinement


cdef class RuleInduction:

    # Functions:

    cdef Rule induce_default_rule(self, uint8[::1, :] y, Loss loss)

    cdef Rule induce_rule(self, intp[::1] nominal_attribute_indices, float32[::1, :] x, intp[::1, :] x_sorted_indices,
                          uint8[::1, :] y, HeadRefinement head_refinement, Loss loss,
                          LabelSubSampling label_sub_sampling, InstanceSubSampling instance_sub_sampling,
                          FeatureSubSampling feature_sub_sampling, Pruning pruning, Shrinkage shrinkage,
                          intp random_state)


cdef class ExactGreedyRuleInduction(RuleInduction):

    # Functions:

    cdef Rule induce_default_rule(self, uint8[::1, :] y, Loss loss)

    cdef Rule induce_rule(self, intp[::1] nominal_attribute_indices, float32[::1, :] x, intp[::1, :] x_sorted_indices,
                          uint8[::1, :] y, HeadRefinement head_refinement, Loss loss,
                          LabelSubSampling label_sub_sampling, InstanceSubSampling instance_sub_sampling,
                          FeatureSubSampling feature_sub_sampling, Pruning pruning, Shrinkage shrinkage,
                          intp random_state)
