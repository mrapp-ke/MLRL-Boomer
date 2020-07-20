from boomer.common._arrays cimport uint32, intp
from boomer.common.rules cimport RuleModel, ModelBuilder
from boomer.common.rule_induction cimport FeatureMatrix, RuleInduction
from boomer.common.head_refinement cimport HeadRefinement
from boomer.common.losses cimport LabelMatrix, Loss
from boomer.common.pruning cimport Pruning
from boomer.common.post_processing cimport PostProcessor
from boomer.common.sub_sampling cimport InstanceSubSampling, FeatureSubSampling, LabelSubSampling


cdef class SequentialRuleInduction:

    # Attributes:

    cdef RuleInduction rule_induction

    cdef HeadRefinement head_refinement

    cdef Loss loss

    cdef list stopping_criteria

    cdef LabelSubSampling label_sub_sampling

    cdef InstanceSubSampling instance_sub_sampling

    cdef FeatureSubSampling feature_sub_sampling

    cdef Pruning pruning

    cdef PostProcessor post_processor

    cdef intp min_coverage

    cdef intp max_conditions

    cdef intp max_head_refinements

    # Functions:

    cpdef RuleModel induce_rules(self, intp[::1] nominal_attribute_indices, FeatureMatrix feature_matrix,
                                 LabelMatrix label_matrix, uint32 random_state, ModelBuilder model_builder)
