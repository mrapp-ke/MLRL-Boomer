from mlrl.common.cython.statistics cimport IStatisticsProviderFactory
from mlrl.common.cython.thresholds cimport IThresholdsFactory
from mlrl.common.cython.rule_induction cimport IRuleInduction
from mlrl.common.cython.head_refinement cimport IHeadRefinementFactory
from mlrl.common.cython.feature_sampling cimport IFeatureSamplingFactory
from mlrl.common.cython.instance_sampling cimport IInstanceSamplingFactory
from mlrl.common.cython.label_sampling cimport ILabelSamplingFactory
from mlrl.common.cython.partition_sampling cimport IPartitionSamplingFactory
from mlrl.common.cython.pruning cimport IPruning
from mlrl.common.cython.post_processing cimport IPostProcessor
from mlrl.common.cython.stopping cimport IStoppingCriterion
from mlrl.common.cython.rule_model_assemblage cimport IRuleModelAssemblage, IRuleModelAssemblageFactory

from libcpp.memory cimport unique_ptr, shared_ptr


cdef extern from "common/algorithm_builder.hpp" nogil:

    cdef cppclass AlgorithmBuilderImpl"AlgorithmBuilder":

        # Constructor:

        AlgorithmBuilderImpl(shared_ptr[IStatisticsProviderFactory] statisticsProviderFactoryPtr,
                             shared_ptr[IThresholdsFactory] thresholdsFactoryPtr,
                             shared_ptr[IRuleInduction] ruleInductionPtr,
                             shared_ptr[IHeadRefinementFactory] headRefinementFactoryPtr,
                             shared_ptr[IRuleModelAssemblageFactory] ruleModelAssemblageFactoryPtr)

        # Functions:

        AlgorithmBuilderImpl& setDefaultRuleHeadRefinementFactory(
            shared_ptr[IHeadRefinementFactory] headRefinementFactoryPtr)

        AlgorithmBuilderImpl& setLabelSamplingFactory(shared_ptr[ILabelSamplingFactory] labelSamplingFactoryPtr)

        AlgorithmBuilderImpl& setInstanceSamplingFactory(
            shared_ptr[IInstanceSamplingFactory] instanceSamplingFactoryPtr)

        AlgorithmBuilderImpl& setFeatureSamplingFactory(shared_ptr[IFeatureSamplingFactory] featureSamplingFactoryPtr)

        AlgorithmBuilderImpl& setPartitionSamplingFactory(
            shared_ptr[IPartitionSamplingFactory] partitionSamplingFactoryPtr)

        AlgorithmBuilderImpl& setPruning(unique_ptr[IPruning] pruningPtr)

        AlgorithmBuilderImpl& setPostProcessor(unique_ptr[IPostProcessor] postProcessorPtr)

        AlgorithmBuilderImpl& addStoppingCriterion(unique_ptr[IStoppingCriterion] stoppingCriterionPtr)

        unique_ptr[IRuleModelAssemblage] build() const


cdef class AlgorithmBuilder:

    # Attributes:

    cdef unique_ptr[AlgorithmBuilderImpl] builder_ptr
