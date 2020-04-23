# distutils: language=c++
from boomer.algorithm._arrays cimport intp, uint8, float32
from boomer.algorithm._model cimport Rule
from boomer.algorithm._losses cimport Loss
from boomer.algorithm._sub_sampling cimport InstanceSubSampling, FeatureSubSampling, LabelSubSampling
from boomer.algorithm._pruning cimport Pruning
from boomer.algorithm._shrinkage cimport Shrinkage
from boomer.algorithm._head_refinement cimport HeadRefinement

from libcpp.unordered_map cimport unordered_map as map


cdef class RuleInduction:

    # Functions:

    cpdef Rule induce_default_rule(self, uint8[::1, :] y, Loss loss)

    cpdef Rule induce_rule(self, intp[::1] nominal_attribute_indices, float32[::1, :] x, uint8[::1, :] y,
                           HeadRefinement head_refinement, Loss loss, LabelSubSampling label_sub_sampling,
                           InstanceSubSampling instance_sub_sampling, FeatureSubSampling feature_sub_sampling,
                           Pruning pruning, Shrinkage shrinkage, random_state: int)


cdef class ExactGreedyRuleInduction(RuleInduction):

    # Attributes:

    cdef map[intp, intp*]* sorted_indices_map_global

    # Functions:

    cpdef Rule induce_default_rule(self, uint8[::1, :] y, Loss loss)

    cpdef Rule induce_rule(self, intp[::1] nominal_attribute_indices, float32[::1, :] x, uint8[::1, :] y,
                           HeadRefinement head_refinement, Loss loss, LabelSubSampling label_sub_sampling,
                           InstanceSubSampling instance_sub_sampling, FeatureSubSampling feature_sub_sampling,
                           Pruning pruning, Shrinkage shrinkage, random_state: int)
