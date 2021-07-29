from mlrl.common.cython._types cimport uint32
from mlrl.common.cython.feature_sampling cimport IFeatureSamplingFactory
from mlrl.common.cython.input cimport NominalFeatureMask, INominalFeatureMask
from mlrl.common.cython.input cimport FeatureMatrix, IFeatureMatrix
from mlrl.common.cython.input cimport LabelMatrix, ILabelMatrix
from mlrl.common.cython.instance_sampling cimport IInstanceSamplingFactory
from mlrl.common.cython.model cimport ModelBuilder, RuleModel, IModelBuilder, RuleModelImpl
from mlrl.common.cython.rule_induction cimport IRuleInduction
from mlrl.common.cython.sampling cimport ILabelSamplingFactory
from mlrl.common.cython.statistics cimport IStatisticsProviderFactory
from mlrl.common.cython.stopping cimport IStoppingCriterion
from mlrl.common.cython.thresholds cimport IThresholdsFactory
from mlrl.common.cython.partition_sampling cimport IPartitionSamplingFactory
from mlrl.common.cython.pruning cimport IPruning
from mlrl.common.cython.post_processing cimport IPostProcessor
from mlrl.common.cython.head_refinement cimport IHeadRefinementFactory

from libcpp.memory cimport unique_ptr, shared_ptr
from libcpp.forward_list cimport forward_list


cdef extern from "common/rule_induction/rule_model_assemblage.hpp" nogil:

    cdef cppclass IRuleModelAssemblage:

        # Functions:

        unique_ptr[RuleModelImpl] induceRules(const INominalFeatureMask& nominalFeatureMask,
                                              const IFeatureMatrix& featureMatrix, const ILabelMatrix& labelMatrix,
                                              uint32 randomState, IModelBuilder& modelBuilder)


cdef extern from "common/rule_induction/rule_model_assemblage_sequential.hpp" nogil:

    cdef cppclass SequentialRuleModelAssemblageImpl"SequentialRuleModelAssemblage"(IRuleModelAssemblage):

        # Constructors:

        SequentialRuleModelAssemblageImpl(
                shared_ptr[IStatisticsProviderFactory] statisticsProviderFactoryPtr,
                shared_ptr[IThresholdsFactory] thresholdsFactoryPtr, shared_ptr[IRuleInduction] ruleInductionPtr,
                shared_ptr[IHeadRefinementFactory] defaultRuleHeadRefinementFactoryPtr,
                shared_ptr[IHeadRefinementFactory] headRefinementFactoryPtr,
                shared_ptr[ILabelSamplingFactory] labelSamplingFactoryPtr,
                shared_ptr[IInstanceSamplingFactory] instanceSamplingFactoryPtr,
                shared_ptr[IFeatureSamplingFactory] featureSamplingFactoryPtr,
                shared_ptr[IPartitionSamplingFactory] partitionSamplingFactoryPtr, shared_ptr[IPruning] pruningPtr,
                shared_ptr[IPostProcessor] postProcessorPtr,
                unique_ptr[forward_list[shared_ptr[IStoppingCriterion]]] stoppingCriteriaPtr)


cdef class RuleModelAssemblage:

    # Attributes:

    cdef shared_ptr[IRuleModelAssemblage] rule_model_assemblage_ptr

    # Functions:

    cpdef RuleModel induce_rules(self, NominalFeatureMask nominal_feature_mask, FeatureMatrix feature_matrix,
                                 LabelMatrix label_matrix, uint32 random_state, ModelBuilder model_builder)



cdef class SequentialRuleModelAssemblage(RuleModelAssemblage):
    pass
