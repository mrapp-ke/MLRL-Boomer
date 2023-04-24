
from mlrl.common.cython.learner cimport IRuleLearner, RuleLearner, ISequentialRuleModelAssemblageMixin, \
    IDefaultRuleMixin, IGreedyTopDownRuleInductionMixin, IBeamSearchTopDownRuleInductionMixin, INoPostProcessorMixin, \
    INoFeatureBinningMixin, IEqualWidthFeatureBinningMixin, IEqualFrequencyFeatureBinningMixin, INoLabelSamplingMixin, \
    ILabelSamplingWithoutReplacementMixin, INoInstanceSamplingMixin, IInstanceSamplingWithoutReplacementMixin, \
    IInstanceSamplingWithReplacementMixin, ILabelWiseStratifiedInstanceSamplingMixin, \
    IExampleWiseStratifiedInstanceSamplingMixin, INoFeatureSamplingMixin, IFeatureSamplingWithoutReplacementMixin, \
    INoPartitionSamplingMixin, IRandomBiPartitionSamplingMixin, ILabelWiseStratifiedBiPartitionSamplingMixin, \
    IExampleWiseStratifiedBiPartitionSamplingMixin, INoRulePruningMixin, IIrepRulePruningMixin, \
    INoParallelRuleRefinementMixin, IParallelRuleRefinementMixin, INoParallelStatisticUpdateMixin, \
    IParallelStatisticUpdateMixin, INoParallelPredictionMixin, IParallelPredictionMixin, \
    INoSizeStoppingCriterionMixin, ISizeStoppingCriterionMixin, INoTimeStoppingCriterionMixin, \
    ITimeStoppingCriterionMixin, IPrePruningMixin, INoGlobalPruningMixin, IPostPruningMixin, \
    INoSequentialPostOptimizationMixin, ISequentialPostOptimizationMixin
from mlrl.boosting.cython.learner cimport IBoostingRuleLearnerConfig, BoostingRuleLearnerConfig, \
    IConstantShrinkageMixin, INoL1RegularizationMixin, IL1RegularizationMixin, INoL2RegularizationMixin, \
    IL2RegularizationMixin, INoDefaultRuleMixin, ICompleteHeadMixin, IDynamicPartialHeadMixin, IFixedPartialHeadMixin, \
    ISingleLabelHeadMixin, IDenseStatisticsMixin, ISparseStatisticsMixin, IExampleWiseLogisticLossMixin, \
    IExampleWiseSquaredErrorLossMixin, IExampleWiseSquaredHingeLossMixin, ILabelWiseLogisticLossMixin, \
    ILabelWiseSquaredErrorLossMixin, ILabelWiseSquaredHingeLossMixin, ILabelBinningMixin, \
    IExampleWiseBinaryPredictorMixin, IGfmBinaryPredictorMixin, IMarginalizedProbabilityPredictorMixin, DdotFunction, \
    DspmvFunction, DsysvFunction

from libcpp.memory cimport unique_ptr


cdef extern from "boosting/learner_boomer.hpp" namespace "boosting" nogil:

    cdef cppclass IBoomerConfig"boosting::IBoomer::IConfig"(IBoostingRuleLearnerConfig,
                                                            IConstantShrinkageMixin,
                                                            INoL1RegularizationMixin,
                                                            IL1RegularizationMixin,
                                                            INoL2RegularizationMixin,
                                                            IL2RegularizationMixin,
                                                            INoDefaultRuleMixin,
                                                            ICompleteHeadMixin,
                                                            IDynamicPartialHeadMixin,
                                                            IFixedPartialHeadMixin,
                                                            ISingleLabelHeadMixin,
                                                            IDenseStatisticsMixin,
                                                            ISparseStatisticsMixin,
                                                            IExampleWiseLogisticLossMixin,
                                                            IExampleWiseSquaredErrorLossMixin,
                                                            IExampleWiseSquaredHingeLossMixin,
                                                            ILabelWiseLogisticLossMixin,
                                                            ILabelWiseSquaredErrorLossMixin,
                                                            ILabelWiseSquaredHingeLossMixin,
                                                            ILabelBinningMixin,
                                                            IExampleWiseBinaryPredictorMixin,
                                                            IGfmBinaryPredictorMixin,
                                                            ISequentialRuleModelAssemblageMixin,
                                                            IDefaultRuleMixin,
                                                            IMarginalizedProbabilityPredictorMixin,
                                                            IGreedyTopDownRuleInductionMixin,
                                                            IBeamSearchTopDownRuleInductionMixin,
                                                            INoPostProcessorMixin,
                                                            INoFeatureBinningMixin,
                                                            IEqualWidthFeatureBinningMixin,
                                                            IEqualFrequencyFeatureBinningMixin,
                                                            INoLabelSamplingMixin,
                                                            ILabelSamplingWithoutReplacementMixin,
                                                            INoInstanceSamplingMixin,
                                                            IInstanceSamplingWithoutReplacementMixin,
                                                            IInstanceSamplingWithReplacementMixin,
                                                            ILabelWiseStratifiedInstanceSamplingMixin,
                                                            IExampleWiseStratifiedInstanceSamplingMixin,
                                                            INoFeatureSamplingMixin,
                                                            IFeatureSamplingWithoutReplacementMixin,
                                                            INoPartitionSamplingMixin,
                                                            IRandomBiPartitionSamplingMixin,
                                                            ILabelWiseStratifiedBiPartitionSamplingMixin,
                                                            IExampleWiseStratifiedBiPartitionSamplingMixin,
                                                            INoRulePruningMixin,
                                                            IIrepRulePruningMixin,
                                                            INoParallelRuleRefinementMixin,
                                                            IParallelRuleRefinementMixin,
                                                            INoParallelStatisticUpdateMixin,
                                                            IParallelStatisticUpdateMixin,
                                                            INoParallelPredictionMixin,
                                                            IParallelPredictionMixin,
                                                            INoSizeStoppingCriterionMixin,
                                                            ISizeStoppingCriterionMixin,
                                                            INoTimeStoppingCriterionMixin,
                                                            ITimeStoppingCriterionMixin,
                                                            IPrePruningMixin,
                                                            INoGlobalPruningMixin,
                                                            IPostPruningMixin,
                                                            INoSequentialPostOptimizationMixin,
                                                            ISequentialPostOptimizationMixin):

        # Functions:

        void useAutomaticDefaultRule()

        void useAutomaticPartitionSampling()

        void useAutomaticFeatureBinning()

        void useAutomaticParallelRuleRefinement()

        void useAutomaticParallelStatisticUpdate()

        void useAutomaticHeads()

        void useAutomaticStatistics()

        void useAutomaticLabelBinning()

        void useAutomaticProbabilityPredictor()


    cdef cppclass IBoomer(IRuleLearner):
        pass


    unique_ptr[IBoomerConfig] createBoomerConfig()


    unique_ptr[IBoomer] createBoomer(unique_ptr[IBoomerConfig] configPtr, DdotFunction ddotFunction,
                                     DspmvFunction dspmvFunction, DsysvFunction dsysvFunction)


cdef class BoomerConfig(BoostingRuleLearnerConfig):

    # Attributes:

    cdef unique_ptr[IBoomerConfig] rule_learner_config_ptr


cdef class Boomer(RuleLearner):

    # Attributes:

    cdef unique_ptr[IBoomer] rule_learner_ptr
