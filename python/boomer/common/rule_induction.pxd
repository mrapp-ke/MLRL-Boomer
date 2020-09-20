from boomer.common._arrays cimport uint32, intp, float32
from boomer.common._tuples cimport IndexedFloat32Array
from boomer.common._random cimport RNG
from boomer.common.rules cimport ModelBuilder
from boomer.common.input_data cimport AbstractFeatureMatrix, AbstractNominalFeatureSet
from boomer.common.statistics cimport StatisticsProvider
from boomer.common.sub_sampling cimport InstanceSubSampling, FeatureSubSampling, LabelSubSampling
from boomer.common.pruning cimport Pruning
from boomer.common.post_processing cimport PostProcessor
from boomer.common.head_refinement cimport AbstractHeadRefinement

from libcpp.unordered_map cimport unordered_map


cdef class RuleInduction:

    # Functions:

    cdef void induce_default_rule(self, StatisticsProvider statistics_provider, AbstractHeadRefinement* head_refinement,
                                  ModelBuilder model_builder)

    cdef bint induce_rule(self, StatisticsProvider statistics_provider, AbstractNominalFeatureSet* nominal_feature_set,
                          AbstractFeatureMatrix* feature_matrix, AbstractHeadRefinement* head_refinement,
                          LabelSubSampling label_sub_sampling, InstanceSubSampling instance_sub_sampling,
                          FeatureSubSampling feature_sub_sampling, Pruning pruning, PostProcessor post_processor,
                          uint32 min_coverage, intp max_conditions, intp max_head_refinements, int num_threads, RNG rng,
                          ModelBuilder model_builder)


cdef class TopDownGreedyRuleInduction(RuleInduction):

    # Attributes:

    cdef unordered_map[uint32, IndexedFloat32Array*]* cache_global

    # Functions:

    cdef void induce_default_rule(self, StatisticsProvider statistics_provider, AbstractHeadRefinement* head_refinement,
                                  ModelBuilder model_builder)

    cdef bint induce_rule(self, StatisticsProvider statistics_provider, AbstractNominalFeatureSet* nominal_feature_set,
                          AbstractFeatureMatrix* feature_matrix, AbstractHeadRefinement* head_refinement,
                          LabelSubSampling label_sub_sampling, InstanceSubSampling instance_sub_sampling,
                          FeatureSubSampling feature_sub_sampling, Pruning pruning, PostProcessor post_processor,
                          uint32 min_coverage, intp max_conditions, intp max_head_refinements, int num_threads, RNG rng,
                          ModelBuilder model_builder)
