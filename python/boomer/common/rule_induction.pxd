from boomer.common._arrays cimport intp, float32
from boomer.common._tuples cimport IndexedFloat32, IndexedFloat32Array
from boomer.common._random cimport RNG
from boomer.common.rules cimport ModelBuilder
from boomer.common.input_data cimport LabelMatrix
from boomer.common.statistics cimport Statistics
from boomer.common.sub_sampling cimport InstanceSubSampling, FeatureSubSampling, LabelSubSampling
from boomer.common.pruning cimport Pruning
from boomer.common.post_processing cimport PostProcessor
from boomer.common.head_refinement cimport HeadRefinement
from boomer.common.rule_evaluation cimport DefaultRuleEvaluation

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


cdef class FeatureMatrix:

    # Attributes:

    cdef readonly intp num_examples

    cdef readonly intp num_features

    # Functions:

    cdef IndexedFloat32Array* get_sorted_feature_values(self, intp feature_index) nogil


cdef class DenseFeatureMatrix(FeatureMatrix):

    # Attributes:

    cdef const float32[::1, :] x

    # Functions:

    cdef IndexedFloat32Array* get_sorted_feature_values(self, intp feature_index) nogil


cdef class CscFeatureMatrix(FeatureMatrix):

    # Attributes:

    cdef const float32[::1] x_data

    cdef const intp[::1] x_row_indices

    cdef const intp[::1] x_col_indices

    # Functions:

    cdef IndexedFloat32Array* get_sorted_feature_values(self, intp feature_index) nogil


cdef class RuleInduction:

    # Functions:

    cdef void induce_default_rule(self, LabelMatrix label_matrix, ModelBuilder model_builder)

    cdef bint induce_rule(self, intp[::1] nominal_attribute_indices, FeatureMatrix feature_matrix, intp num_labels,
                          HeadRefinement head_refinement, LabelSubSampling label_sub_sampling,
                          InstanceSubSampling instance_sub_sampling, FeatureSubSampling feature_sub_sampling,
                          Pruning pruning, PostProcessor post_processor, intp min_coverage, intp max_conditions,
                          intp max_head_refinements, RNG rng, ModelBuilder model_builder)


cdef class ExactGreedyRuleInduction(RuleInduction):

    # Attributes:

    cdef DefaultRuleEvaluation default_rule_evaluation

    cdef Statistics statistics

    cdef unordered_map[intp, IndexedFloat32Array*]* cache_global

    # Functions:

    cdef void induce_default_rule(self, LabelMatrix label_matrix, ModelBuilder model_builder)

    cdef bint induce_rule(self, intp[::1] nominal_attribute_indices, FeatureMatrix feature_matrix, intp num_labels,
                          HeadRefinement head_refinement, LabelSubSampling label_sub_sampling,
                          InstanceSubSampling instance_sub_sampling, FeatureSubSampling feature_sub_sampling,
                          Pruning pruning, PostProcessor post_processor, intp min_coverage, intp max_conditions,
                          intp max_head_refinements, RNG rng, ModelBuilder model_builder)
