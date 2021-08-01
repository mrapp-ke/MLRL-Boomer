from mlrl.common.cython.statistics cimport StatisticsProviderFactory
from mlrl.common.cython.thresholds cimport ThresholdsFactory
from mlrl.common.cython.rule_induction cimport RuleInduction
from mlrl.common.cython.head_refinement cimport HeadRefinementFactory
from mlrl.common.cython.rule_model_assemblage cimport RuleModelAssemblageFactory

from libcpp.utility cimport move
from libcpp.memory cimport make_unique


cdef class AlgorithmBuilder:
    """
    A wrapper for the C++ class `AlgorithmBuilder`.
    """

    def __cinit__(self, StatisticsProviderFactory statistics_provider_factory, ThresholdsFactory thresholds_factory,
                  RuleInduction rule_induction, HeadRefinementFactory default_rule_head_refinement_factory,
                  HeadRefinementFactory regular_rule_head_refinement_factory,
                  RuleModelAssemblageFactory rule_model_assemblage_factory):
        """
        :param statistics_provider_factory:             TODO
        :param thresholds_factory:                      TODO
        :param rule_induction:                          TODO
        :param default_rule_head_refinement_factory:    TODO
        :param regular_rule_head_refinement_factory:    TODO
        :param rule_model_assemblage_factory:           TODO
        """
        self.builder_ptr = make_unique[AlgorithmBuilderImpl](
            statistics_provider_factory.statistics_provider_factory_ptr, thresholds_factory.thresholds_factory_ptr,
            rule_induction.rule_induction_ptr, default_rule_head_refinement_factory.head_refinement_factory_ptr,
            regular_rule_head_refinement_factory.head_refinement_factory_ptr,
            rule_model_assemblage_factory.rule_model_assemblage_factory_ptr)

    cpdef AlgorithmBuilder set_label_sampling_factory(self, LabelSamplingFactory label_sampling_factory):
        self.builder_ptr.get().setLabelSamplingFactory(label_sampling_factory.label_sampling_factory_ptr)
        return self

    cpdef AlgorithmBuilder set_instance_sampling_factory(self, InstanceSamplingFactory instance_sampling_factory):
        self.builder_ptr.get().setInstanceSamplingFactory(instance_sampling_factory.instance_sampling_factory_ptr)
        return self

    cpdef AlgorithmBuilder set_feature_sampling_factory(self, FeatureSamplingFactory feature_sampling_factory):
        self.builder_ptr.get().setFeatureSamplingFactory(feature_sampling_factory.feature_sampling_factory_ptr)
        return self

    cpdef AlgorithmBuilder set_partition_sampling_factory(self, PartitionSamplingFactory partition_sampling_factory):
        self.builder_ptr.get().setPartitionSamplingFactory(partition_sampling_factory.partition_sampling_factory_ptr)
        return self

    cpdef AlgorithmBuilder set_pruning(self, Pruning pruning):
        self.builder_ptr.get().setPruning(pruning.pruning_ptr)
        return self

    cpdef AlgorithmBuilder set_post_processor(self, PostProcessor post_processor):
        self.builder_ptr.get().setPostProcessor(post_processor.post_processor_ptr)
        return self

    cpdef AlgorithmBuilder add_stopping_criterion(self, StoppingCriterion stopping_criterion):
        self.builder_ptr.get().addStoppingCriterion(stopping_criterion.stopping_criterion_ptr)
        return self

    cpdef RuleModelAssemblage build(self):
        cdef RuleModelAssemblage rule_model_assemblage = RuleModelAssemblage.__new__(RuleModelAssemblage)
        rule_model_assemblage.rule_model_assemblage_ptr = move(self.builder_ptr.get().build())
        return rule_model_assemblage
