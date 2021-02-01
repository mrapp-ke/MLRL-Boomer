from boomer.common._types cimport uint32, intp
from boomer.common.input cimport NominalFeatureMask, INominalFeatureMask
from boomer.common.input cimport FeatureMatrix, IFeatureMatrix
from boomer.common.input cimport LabelMatrix, ILabelMatrix
from boomer.common.model cimport ModelBuilder, RuleModel, IModelBuilder, RuleModelImpl
from boomer.common.sampling cimport ILabelSubSampling, IInstanceSubSampling, IFeatureSubSampling, RNG
from boomer.common.statistics cimport IStatisticsProviderFactory
from boomer.common.stopping cimport IStoppingCriterion
from boomer.common.thresholds cimport IThresholdsFactory
from boomer.common.pruning cimport IPruning
from boomer.common.post_processing cimport IPostProcessor
from boomer.common.head_refinement cimport IHeadRefinementFactory

from libcpp cimport bool
from libcpp.memory cimport unique_ptr, shared_ptr
from libcpp.forward_list cimport forward_list


cdef extern from "cpp/rule_induction/rule_induction.h" nogil:

    cdef cppclass IRuleInduction:
        pass


cdef extern from "cpp/rule_induction/rule_model_induction.h" nogil:

    cdef cppclass IRuleModelInduction:

        # Functions:

        unique_ptr[RuleModelImpl] induceRules(shared_ptr[INominalFeatureMask] nominalFeatureMaskPtr,
                                              shared_ptr[IFeatureMatrix] featureMatrixPtr,
                                              shared_ptr[ILabelMatrix] labelMatrixPtr, RNG& rng,
                                              IModelBuilder& modelBuilder)


cdef extern from "cpp/rule_induction/rule_induction_top_down.h" nogil:

    cdef cppclass TopDownRuleInductionImpl"TopDownRuleInduction"(IRuleInduction):

        # Constructors:

        TopDownRuleInductionImpl(uint32 numThreads) except +


cdef extern from "cpp/rule_induction/rule_model_induction_sequential.h" nogil:

    cdef cppclass SequentialRuleModelInductionImpl"SequentialRuleModelInduction"(IRuleModelInduction):

        # Constructors:

        SequentialRuleModelInductionImpl(shared_ptr[IStatisticsProviderFactory] statisticsProviderFactoryPtr,
                                         shared_ptr[IThresholdsFactory] thresholdsFactoryPtr,
                                         shared_ptr[IRuleInduction] ruleInductionPtr,
                                         shared_ptr[IHeadRefinementFactory] defaultRuleHeadRefinementFactoryPtr,
                                         shared_ptr[IHeadRefinementFactory] headRefinementFactoryPtr,
                                         shared_ptr[ILabelSubSampling] labelSubSamplingPtr,
                                         shared_ptr[IInstanceSubSampling] instanceSubSamplingPtr,
                                         shared_ptr[IFeatureSubSampling] featureSubSamplingPtr,
                                         shared_ptr[IPruning] pruningPtr, shared_ptr[IPostProcessor] postProcessorPtr,
                                         uint32 minCoverage, intp maxConditions, intp maxHeadRefinements,
                                         unique_ptr[forward_list[shared_ptr[IStoppingCriterion]]] stoppingCriteriaPtr) except +


cdef class RuleInduction:

    # Attributes:

    cdef shared_ptr[IRuleInduction] rule_induction_ptr


cdef class TopDownRuleInduction(RuleInduction):
    pass


cdef class RuleModelInduction:

    # Attributes:

    cdef shared_ptr[IRuleModelInduction] rule_model_induction_ptr

    # Functions:

    cpdef RuleModel induce_rules(self, NominalFeatureMask nominal_feature_mask, FeatureMatrix feature_matrix,
                                 LabelMatrix label_matrix, uint32 random_state, ModelBuilder model_builder)



cdef class SequentialRuleModelInduction(RuleModelInduction):
    pass
