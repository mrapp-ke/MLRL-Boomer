from boomer.common._arrays cimport uint8, intp, float32
from boomer.common._tuples cimport IndexedFloat32, IndexedFloat32Array
from boomer.common._random cimport RNG
from boomer.common.rules cimport ModelBuilder
from boomer.common.input_data cimport RandomAccessLabelMatrix, FeatureMatrix
from boomer.common.statistics cimport AbstractStatistics
from boomer.common.sub_sampling cimport InstanceSubSampling, FeatureSubSampling, LabelSubSampling
from boomer.common.pruning cimport Pruning
from boomer.common.post_processing cimport PostProcessor
from boomer.common.head_refinement cimport HeadRefinement
from boomer.common.rule_evaluation cimport AbstractDefaultRuleEvaluation

from libcpp.memory cimport shared_ptr
from libcpp.unordered_map cimport unordered_map


"""
A struct that contains a pointer to a struct of type `IndexedFloat32Array`, representing the indices and feature values
of the training examples that are covered by a rule. The attribute `num_conditions` specifies how many conditions the
rule contained when the array was updated for the last time. It may be used to check if the array is still valid or must
be updated.
"""
cdef struct IndexedFloat32ArrayWrapper:
    IndexedFloat32Array* array
    intp num_conditions


cdef class RuleInduction:

    # Functions:

    cdef void induce_default_rule(self, RandomAccessLabelMatrix label_matrix, ModelBuilder model_builder)

    cdef bint induce_rule(self, uint8[::1] nominal_attribute_mask, FeatureMatrix feature_matrix, intp num_labels,
                          HeadRefinement head_refinement, LabelSubSampling label_sub_sampling,
                          InstanceSubSampling instance_sub_sampling, FeatureSubSampling feature_sub_sampling,
                          Pruning pruning, PostProcessor post_processor, intp min_coverage, intp max_conditions,
                          intp max_head_refinements, int num_threads, RNG rng, ModelBuilder model_builder)


cdef class ExactGreedyRuleInduction(RuleInduction):

    # Attributes:

    cdef shared_ptr[AbstractDefaultRuleEvaluation] default_rule_evaluation_ptr

    cdef shared_ptr[AbstractStatistics] statistics_ptr

    cdef unordered_map[intp, IndexedFloat32Array*]* cache_global

    # Functions:

    cdef void induce_default_rule(self, RandomAccessLabelMatrix label_matrix, ModelBuilder model_builder)

    cdef bint induce_rule(self, uint8[::1] nominal_attribute_mask, FeatureMatrix feature_matrix, intp num_labels,
                          HeadRefinement head_refinement, LabelSubSampling label_sub_sampling,
                          InstanceSubSampling instance_sub_sampling, FeatureSubSampling feature_sub_sampling,
                          Pruning pruning, PostProcessor post_processor, intp min_coverage, intp max_conditions,
                          intp max_head_refinements, int num_threads, RNG rng, ModelBuilder model_builder)
