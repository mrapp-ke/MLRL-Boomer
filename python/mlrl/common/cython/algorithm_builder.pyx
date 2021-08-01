"""
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)
"""
from mlrl.common.cython.feature_sampling cimport FeatureSamplingFactory
from mlrl.common.cython.head_refinement cimport HeadRefinementFactory
from mlrl.common.cython.instance_sampling cimport InstanceSamplingFactory
from mlrl.common.cython.label_sampling cimport LabelSamplingFactory
from mlrl.common.cython.partition_sampling cimport PartitionSamplingFactory
from mlrl.common.cython.pruning cimport Pruning
from mlrl.common.cython.post_processing cimport PostProcessor
from mlrl.common.cython.rule_induction cimport RuleInduction
from mlrl.common.cython.rule_model_assemblage cimport RuleModelAssemblage, RuleModelAssemblageFactory
from mlrl.common.cython.statistics cimport StatisticsProviderFactory
from mlrl.common.cython.stopping cimport StoppingCriterion
from mlrl.common.cython.thresholds cimport ThresholdsFactory

from libcpp.utility cimport move
from libcpp.memory cimport make_unique


cdef class AlgorithmBuilder:
    """
    A wrapper for the C++ class `AlgorithmBuilder`.
    """

    def __cinit__(self, StatisticsProviderFactory statistics_provider_factory not None,
                  ThresholdsFactory thresholds_factory not None, RuleInduction rule_induction not None,
                  HeadRefinementFactory head_refinement_factory not None,
                  RuleModelAssemblageFactory rule_model_assemblage_factory not None):
        """
        :param statistics_provider_factory:     TODO
        :param thresholds_factory:              TODO
        :param rule_induction:                  TODO
        :param head_refinement_factory:         TODO
        :param rule_model_assemblage_factory:   TODO
        """
        self.builder_ptr = make_unique[AlgorithmBuilderImpl](
            statistics_provider_factory.statistics_provider_factory_ptr, thresholds_factory.thresholds_factory_ptr,
            rule_induction.rule_induction_ptr, head_refinement_factory.head_refinement_factory_ptr,
            rule_model_assemblage_factory.rule_model_assemblage_factory_ptr)

    def set_default_rule_head_refinement_factory(
            self, HeadRefinementFactory head_refinement_factory not None) -> AlgorithmBuilder:
        """
        TODO

        :param head_refinement_factory: TODO
        :return:                        TODO
        """
        self.builder_ptr.get().setDefaultRuleHeadRefinementFactory(head_refinement_factory.head_refinement_factory_ptr)
        return self

    def set_label_sampling_factory(self, LabelSamplingFactory label_sampling_factory not None) -> AlgorithmBuilder:
        """
        TODO

        :param label_sampling_factory:  TODO
        :return:                        TODO
        """
        self.builder_ptr.get().setLabelSamplingFactory(label_sampling_factory.label_sampling_factory_ptr)
        return self

    def set_instance_sampling_factory(self,
                                      InstanceSamplingFactory instance_sampling_factory not None) -> AlgorithmBuilder:
        """
        TODO

        :param instance_sampling_factory:   TODO
        :return:                            TODO
        """
        self.builder_ptr.get().setInstanceSamplingFactory(instance_sampling_factory.instance_sampling_factory_ptr)
        return self

    def set_feature_sampling_factory(self,
                                     FeatureSamplingFactory feature_sampling_factory not None) -> AlgorithmBuilder:
        """
        TODO

        :param feature_sampling_factory:    TODO
        :return:                            TODO
        """
        self.builder_ptr.get().setFeatureSamplingFactory(feature_sampling_factory.feature_sampling_factory_ptr)
        return self

    def set_partition_sampling_factory(
            self, PartitionSamplingFactory partition_sampling_factory not None) -> AlgorithmBuilder:
        """
        TODO

        :param partition_sampling_factory:  TODO
        :return:                            TODO
        """
        self.builder_ptr.get().setPartitionSamplingFactory(move(partition_sampling_factory.partition_sampling_factory_ptr))
        return self

    def set_pruning(self, Pruning pruning not None ) -> AlgorithmBuilder:
        """
        TODO

        :param pruning: TODO
        :return:        TODO
        """
        self.builder_ptr.get().setPruning(move(pruning.pruning_ptr))
        return self

    def set_post_processor(self, PostProcessor post_processor not None) -> AlgorithmBuilder:
        """
        TODO

        :param post_processor:  TODO
        :return:                TODO
        """
        self.builder_ptr.get().setPostProcessor(move(post_processor.post_processor_ptr))
        return self

    def add_stopping_criterion(self, StoppingCriterion stopping_criterion not None) -> AlgorithmBuilder:
        """
        TODO

        :param stopping_criterion:  TODO
        :return:                    TODO
        """
        self.builder_ptr.get().addStoppingCriterion(move(stopping_criterion.stopping_criterion_ptr))
        return self

    def build(self) -> RuleModelAssemblage:
        """
        TODO

        :return: TODO
        """
        cdef RuleModelAssemblage rule_model_assemblage = RuleModelAssemblage.__new__(RuleModelAssemblage)
        rule_model_assemblage.rule_model_assemblage_ptr = move(self.builder_ptr.get().build())
        return rule_model_assemblage
