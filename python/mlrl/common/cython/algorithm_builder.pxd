from mlrl.common.cython.statistics cimport IStatisticsProviderFactory
from mlrl.common.cython.thresholds cimport IThresholdsFactory
from mlrl.common.cython.rule_induction cimport IRuleInduction
from mlrl.common.cython.head_refinement cimport IHeadRefinementFactory
from mlrl.common.cython.feature_sampling cimport FeatureSamplingFactory, IFeatureSamplingFactory
from mlrl.common.cython.instance_sampling cimport InstanceSamplingFactory, IInstanceSamplingFactory
from mlrl.common.cython.label_sampling cimport LabelSamplingFactory, ILabelSamplingFactory
from mlrl.common.cython.partition_sampling cimport PartitionSamplingFactory, IPartitionSamplingFactory
from mlrl.common.cython.pruning cimport Pruning, IPruning
from mlrl.common.cython.post_processing cimport PostProcessor, IPostProcessor
from mlrl.common.cython.stopping cimport StoppingCriterion, IStoppingCriterion
from mlrl.common.cython.rule_model_assemblage cimport RuleModelAssemblage, IRuleModelAssemblage, \
    IRuleModelAssemblageFactory

from libcpp.memory cimport unique_ptr, shared_ptr


cdef extern from "common/algorithm_builder.hpp" nogil:

    cdef cppclass AlgorithmBuilderImpl"AlgorithmBuilder":

        # Constructor:

        AlgorithmBuilderImpl(shared_ptr[IStatisticsProviderFactory] statisticsProviderFactoryPtr,
                             shared_ptr[IThresholdsFactory] thresholdsFactoryPtr,
                             shared_ptr[IRuleInduction] ruleInductionPtr,
                             shared_ptr[IHeadRefinementFactory] defaultRuleHeadRefinementFactoryPtr,
                             shared_ptr[IHeadRefinementFactory] regularRuleHeadRefinementFactoryPtr,
                             shared_ptr[IRuleModelAssemblageFactory] ruleModelAssemblageFactoryPtr)

        # Functions:

        AlgorithmBuilderImpl& setLabelSamplingFactory(shared_ptr[ILabelSamplingFactory] labelSamplingFactoryPtr)

        AlgorithmBuilderImpl& setInstanceSamplingFactory(
            shared_ptr[IInstanceSamplingFactory] instanceSamplingFactoryPtr)

        AlgorithmBuilderImpl& setFeatureSamplingFactory(shared_ptr[IFeatureSamplingFactory] featureSamplingFactoryPtr)

        AlgorithmBuilderImpl& setPartitionSamplingFactory(
            shared_ptr[IPartitionSamplingFactory] partitionSamplingFactoryPtr)

        AlgorithmBuilderImpl& setPruning(shared_ptr[IPruning] pruningPtr)

        AlgorithmBuilderImpl& setPostProcessor(shared_ptr[IPostProcessor] postProcessorPtr)

        AlgorithmBuilderImpl& addStoppingCriterion(shared_ptr[IStoppingCriterion] stoppingCriterionPtr)

        unique_ptr[IRuleModelAssemblage] build() const


cdef class AlgorithmBuilder:

    # Attributes:

    cdef unique_ptr[AlgorithmBuilderImpl] builder_ptr

    # Functions:

    cpdef AlgorithmBuilder set_label_sampling_factory(self, LabelSamplingFactory label_sampling_factory)

    cpdef AlgorithmBuilder set_instance_sampling_factory(self, InstanceSamplingFactory instance_sampling_factory)

    cpdef AlgorithmBuilder set_feature_sampling_factory(self, FeatureSamplingFactory feature_sampling_factory)

    cpdef AlgorithmBuilder set_partition_sampling_factory(self, PartitionSamplingFactory partition_sampling_factory)

    cpdef AlgorithmBuilder set_pruning(self, Pruning pruning)

    cpdef AlgorithmBuilder set_post_processor(self, PostProcessor postProcessor)

    cpdef AlgorithmBuilder add_stopping_criterion(self, StoppingCriterion stoppingCriterion)

    cpdef RuleModelAssemblage build(self)
