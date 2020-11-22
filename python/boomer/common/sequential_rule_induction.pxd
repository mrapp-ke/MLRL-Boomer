from boomer.common._types cimport uint32, intp
from boomer.common.rules cimport RuleModel, ModelBuilder
from boomer.common.rule_induction cimport RuleInduction
from boomer.common.statistics cimport StatisticsProviderFactory
from boomer.common.thresholds cimport ThresholdsFactory
from boomer.common.head_refinement cimport HeadRefinementFactory
from boomer.common.input_data cimport LabelMatrix, FeatureMatrix, NominalFeatureMask
from boomer.common.pruning cimport Pruning
from boomer.common.post_processing cimport PostProcessor
from boomer.common.sampling cimport InstanceSubSampling, FeatureSubSampling, LabelSubSampling


cdef class SequentialRuleInduction:

    # Attributes:

    cdef StatisticsProviderFactory statistics_provider_factory

    cdef ThresholdsFactory thresholds_factory

    cdef RuleInduction rule_induction

    cdef HeadRefinementFactory default_rule_head_refinement_factory

    cdef HeadRefinementFactory head_refinement_factory

    cdef list stopping_criteria

    cdef LabelSubSampling label_sub_sampling

    cdef InstanceSubSampling instance_sub_sampling

    cdef FeatureSubSampling feature_sub_sampling

    cdef Pruning pruning

    cdef PostProcessor post_processor

    cdef uint32 min_coverage

    cdef intp max_conditions

    cdef intp max_head_refinements

    cdef int num_threads

    # Functions:

    cpdef RuleModel induce_rules(self, NominalFeatureMask nominal_feature_mask, FeatureMatrix feature_matrix,
                                 LabelMatrix label_matrix, uint32 random_state, ModelBuilder model_builder)
