"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)
"""
from mlrl.common.cython.feature_sampling cimport FeatureSamplingFactory
from mlrl.common.cython.head_refinement cimport HeadRefinementFactory
from mlrl.common.cython.instance_sampling cimport InstanceSamplingFactory
from mlrl.common.cython.post_processing cimport PostProcessor
from mlrl.common.cython.pruning cimport Pruning
from mlrl.common.cython.partition_sampling cimport PartitionSamplingFactory
from mlrl.common.cython.rule_induction cimport RuleInduction
from mlrl.common.cython.label_sampling cimport LabelSamplingFactory
from mlrl.common.cython.statistics cimport StatisticsProviderFactory
from mlrl.common.cython.stopping cimport StoppingCriterion
from mlrl.common.cython.thresholds cimport ThresholdsFactory

from cython.operator cimport dereference

from libcpp.memory cimport make_unique, make_shared
from libcpp.utility cimport move


cdef class RuleModelAssemblage:
    """
    A wrapper for the pure virtual C++ class `IRuleModelAssemblage`.
    """

    cpdef RuleModel induce_rules(self, NominalFeatureMask nominal_feature_mask, FeatureMatrix feature_matrix,
                                 LabelMatrix label_matrix, uint32 random_state, ModelBuilder model_builder):
        cdef shared_ptr[IRuleModelAssemblage] rule_model_assemblage_ptr = self.rule_model_assemblage_ptr
        cdef unique_ptr[RuleModelImpl] rule_model_ptr = rule_model_assemblage_ptr.get().induceRules(
            dereference(nominal_feature_mask.nominal_feature_mask_ptr), dereference(feature_matrix.feature_matrix_ptr),
            dereference(label_matrix.label_matrix_ptr), random_state,
            dereference(model_builder.model_builder_ptr))
        cdef RuleModel model = RuleModel.__new__(RuleModel)
        model.model_ptr = move(rule_model_ptr)
        return model


cdef class SequentialRuleModelAssemblageFactory(RuleModelAssemblageFactory):
    """
    A wrapper for the C++ class `SequentialRuleModelAssemblageFactory`.
    """

    def __cinit__(self):
        self.rule_model_assemblage_factory_ptr = <shared_ptr[IRuleModelAssemblageFactory]>make_shared[SequentialRuleModelAssemblageFactoryImpl]()


cdef class SequentialRuleModelAssemblage(RuleModelAssemblage):
    """
    A wrapper for the C++ class `SequentialRuleModelAssemblage`.
    """

    def __cinit__(self, StatisticsProviderFactory statistics_provider_factory, ThresholdsFactory thresholds_factory,
                  RuleInduction rule_induction, HeadRefinementFactory default_rule_head_refinement_factory,
                  HeadRefinementFactory head_refinement_factory, LabelSamplingFactory label_sampling_factory,
                  InstanceSamplingFactory instance_sampling_factory, FeatureSamplingFactory feature_sampling_factory,
                  PartitionSamplingFactory partition_sampling_factory, Pruning pruning, PostProcessor post_processor,
                  list stopping_criteria):
        """
        :param statistics_provider_factory:             A factory that allows to create a provider that provides access
                                                        to the statistics which serve as the basis for learning rules
        :param thresholds_factory:                      A factory that allows to create objects that provide access to
                                                        the thresholds that may be used by the conditions of rules
        :param rule_induction:                          The algorithm that should be used to induce rules
        :param default_rule_head_refinement_factory:    The factory that allows to create instances of the class that
                                                        implements the strategy that should be used to find the head of
                                                        the default rule
        :param head_refinement_factory:                 The factory that allows to create instances of the class that
                                                        implements the strategy that should be used to find the heads of
                                                        rules
        :param label_sampling_factory:                  The factory that should be used for creating the implementation
                                                        to be used for sampling the labels each time a new
                                                        classification rule is learned
        :param instance_sampling_factory:               The factory that should be used for creating the implementation
                                                        to be used for sampling the training examples each time a new
                                                        classification rule is learned
        :param feature_sampling_factory:                The factory that should be used for creating the implementation
                                                        to be used for sampling the features each time a classification
                                                        rule is refined
        :param partition_sampling_factory:              The factory that should be used for creating the implementation
                                                        to be used for partitioning the training examples into a
                                                        training set and a holdout set
        :param pruning:                                 The strategy that should be used for pruning rules
        :param post_processor:                          The post-processor that should be used to post-process the rule
                                                        once it has been learned
        :param stopping_criteria                        A list that contains the stopping criteria that should be used
                                                        to decide whether additional rules should be induced or not
        """

        cdef unique_ptr[forward_list[shared_ptr[IStoppingCriterion]]] stopping_criteria_ptr = make_unique[forward_list[shared_ptr[IStoppingCriterion]]]()
        cdef uint32 num_stopping_criteria = len(stopping_criteria)
        cdef StoppingCriterion stopping_criterion
        cdef uint32 i

        for i in range(num_stopping_criteria):
            stopping_criterion = stopping_criteria[i]
            stopping_criteria_ptr.get().push_front(stopping_criterion.stopping_criterion_ptr)

        self.rule_model_assemblage_ptr = <shared_ptr[IRuleModelAssemblage]>make_shared[SequentialRuleModelAssemblageImpl](
            statistics_provider_factory.statistics_provider_factory_ptr, thresholds_factory.thresholds_factory_ptr,
            rule_induction.rule_induction_ptr, default_rule_head_refinement_factory.head_refinement_factory_ptr,
            head_refinement_factory.head_refinement_factory_ptr, label_sampling_factory.label_sampling_factory_ptr,
            instance_sampling_factory.instance_sampling_factory_ptr,
            feature_sampling_factory.feature_sampling_factory_ptr,
            partition_sampling_factory.partition_sampling_factory_ptr, pruning.pruning_ptr,
            post_processor.post_processor_ptr, move(stopping_criteria_ptr))
