from boomer.common._types cimport uint32, intp
from boomer.common.rules cimport ModelBuilder
from boomer.common.input_data cimport IFeatureMatrix, INominalFeatureMask
from boomer.common.statistics cimport StatisticsProvider
from boomer.common.thresholds cimport AbstractThresholds
from boomer.common.sampling cimport IInstanceSubSampling, IFeatureSubSampling, ILabelSubSampling, RNG
from boomer.common.pruning cimport IPruning
from boomer.common.post_processing cimport IPostProcessor
from boomer.common.head_refinement cimport IHeadRefinementFactory


cdef class RuleInduction:

    # Functions:

    cdef void induce_default_rule(self, StatisticsProvider statistics_provider,
                                  IHeadRefinementFactory* head_refinement_factory, ModelBuilder model_builder)

    cdef bint induce_rule(self, AbstractThresholds* thresholds, INominalFeatureMask* nominal_feature_mask,
                          IFeatureMatrix* feature_matrix, ILabelSubSampling* label_sub_sampling,
                          IInstanceSubSampling* instance_sub_sampling, IFeatureSubSampling* feature_sub_sampling,
                          IPruning* pruning, IPostProcessor* post_processor, uint32 min_coverage, intp max_conditions,
                          intp max_head_refinements, int num_threads, RNG* rng, ModelBuilder model_builder)


cdef class TopDownGreedyRuleInduction(RuleInduction):

    # Functions:

    cdef void induce_default_rule(self, StatisticsProvider statistics_provider,
                                  IHeadRefinementFactory* head_refinement_factory, ModelBuilder model_builder)

    cdef bint induce_rule(self, AbstractThresholds* thresholds, INominalFeatureMask* nominal_feature_mask,
                          IFeatureMatrix* feature_matrix, ILabelSubSampling* label_sub_sampling,
                          IInstanceSubSampling* instance_sub_sampling, IFeatureSubSampling* feature_sub_sampling,
                          IPruning* pruning, IPostProcessor* post_processor, uint32 min_coverage, intp max_conditions,
                          intp max_head_refinements, int num_threads, RNG* rng, ModelBuilder model_builder)
