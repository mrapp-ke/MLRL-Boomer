from boomer.common._types cimport uint32, intp
from boomer.common._indices cimport IIndexVector
from boomer.common.input cimport IFeatureMatrix, INominalFeatureMask
from boomer.common.model cimport IModelBuilder
from boomer.common.statistics cimport IStatisticsProvider
from boomer.common.thresholds cimport IThresholds
from boomer.common.sampling cimport IWeightVector, IFeatureSubSampling, RNG
from boomer.common.pruning cimport IPruning
from boomer.common.post_processing cimport IPostProcessor
from boomer.common.head_refinement cimport IHeadRefinementFactory


cdef class RuleInduction:

    # Functions:

    cdef void induce_default_rule(self, IStatisticsProvider* statistics_provider,
                                  IHeadRefinementFactory* head_refinement_factory, IModelBuilder* model_builder)

    cdef bint induce_rule(self, IThresholds* thresholds, INominalFeatureMask* nominal_feature_mask,
                          IFeatureMatrix* feature_matrix, IIndexVector* label_indices, IWeightVector* weight_vector,
                          IFeatureSubSampling* feature_sub_sampling, IPruning* pruning, IPostProcessor* post_processor,
                          uint32 min_coverage, intp max_conditions, intp max_head_refinements, int num_threads,
                          RNG* rng, IModelBuilder* model_builder)


cdef class TopDownGreedyRuleInduction(RuleInduction):

    # Functions:

    cdef void induce_default_rule(self, IStatisticsProvider* statistics_provider,
                                  IHeadRefinementFactory* head_refinement_factory, IModelBuilder* model_builder)

    cdef bint induce_rule(self, IThresholds* thresholds, INominalFeatureMask* nominal_feature_mask,
                          IFeatureMatrix* feature_matrix, IIndexVector* label_indices, IWeightVector* weight_vector,
                          IFeatureSubSampling* feature_sub_sampling, IPruning* pruning, IPostProcessor* post_processor,
                          uint32 min_coverage, intp max_conditions, intp max_head_refinements, int num_threads,
                          RNG* rng, IModelBuilder* model_builder)
