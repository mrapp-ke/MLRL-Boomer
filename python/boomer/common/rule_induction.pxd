from boomer.common._arrays cimport uint32, intp
from boomer.common._random cimport RNG
from boomer.common.rules cimport ModelBuilder
from boomer.common.input_data cimport IFeatureMatrix, INominalFeatureVector
from boomer.common.statistics cimport StatisticsProvider
from boomer.common.thresholds cimport AbstractThresholds
from boomer.common.sub_sampling cimport IInstanceSubSampling, IFeatureSubSampling, ILabelSubSampling
from boomer.common.pruning cimport Pruning
from boomer.common.post_processing cimport PostProcessor
from boomer.common.head_refinement cimport IHeadRefinement


cdef class RuleInduction:

    # Functions:

    cdef void induce_default_rule(self, StatisticsProvider statistics_provider, IHeadRefinement* head_refinement,
                                  ModelBuilder model_builder)

    cdef bint induce_rule(self, StatisticsProvider statistics_provider, AbstractThresholds* thresholds,
                          INominalFeatureVector* nominal_feature_vector, IFeatureMatrix* feature_matrix,
                          IHeadRefinement* head_refinement, ILabelSubSampling* label_sub_sampling,
                          IInstanceSubSampling* instance_sub_sampling, IFeatureSubSampling* feature_sub_sampling,
                          Pruning pruning, PostProcessor post_processor, uint32 min_coverage, intp max_conditions,
                          intp max_head_refinements, int num_threads, RNG* rng, ModelBuilder model_builder)


cdef class TopDownGreedyRuleInduction(RuleInduction):

    # Functions:

    cdef void induce_default_rule(self, StatisticsProvider statistics_provider, IHeadRefinement* head_refinement,
                                  ModelBuilder model_builder)

    cdef bint induce_rule(self, StatisticsProvider statistics_provider, AbstractThresholds* thresholds,
                          INominalFeatureVector* nominal_feature_vector, IFeatureMatrix* feature_matrix,
                          IHeadRefinement* head_refinement, ILabelSubSampling* label_sub_sampling,
                          IInstanceSubSampling* instance_sub_sampling, IFeatureSubSampling* feature_sub_sampling,
                          Pruning pruning, PostProcessor post_processor, uint32 min_coverage, intp max_conditions,
                          intp max_head_refinements, int num_threads, RNG* rng, ModelBuilder model_builder)
