
from mlrl.common.cython.learner cimport IRuleLearner, RuleLearner, RuleLearnerConfig, \
    ISequentialRuleModelAssemblageMixin, IDefaultRuleMixin, IGreedyTopDownRuleInductionMixin, \
    IBeamSearchTopDownRuleInductionMixin, INoPostProcessorMixin, INoFeatureBinningMixin, \
    IEqualWidthFeatureBinningMixin, IEqualFrequencyFeatureBinningMixin, INoLabelSamplingMixin, \
    ILabelSamplingWithoutReplacementMixin, INoInstanceSamplingMixin, IInstanceSamplingWithoutReplacementMixin, \
    IInstanceSamplingWithReplacementMixin, ILabelWiseStratifiedInstanceSamplingMixin, \
    IExampleWiseStratifiedInstanceSamplingMixin, INoFeatureSamplingMixin, IFeatureSamplingWithoutReplacementMixin, \
    INoPartitionSamplingMixin, IRandomBiPartitionSamplingMixin, ILabelWiseStratifiedBiPartitionSamplingMixin, \
    IExampleWiseStratifiedBiPartitionSamplingMixin, INoRulePruningMixin, IIrepRulePruningMixin, \
    INoParallelRuleRefinementMixin, IParallelRuleRefinementMixin, INoParallelStatisticUpdateMixin, \
    IParallelStatisticUpdateMixin, INoParallelPredictionMixin, IParallelPredictionMixin, \
    INoSizeStoppingCriterionMixin, ISizeStoppingCriterionMixin, INoTimeStoppingCriterionMixin, \
    ITimeStoppingCriterionMixin, IPrePruningMixin, INoGlobalPruningMixin, IPostPruningMixin, \
    INoSequentialPostOptimizationMixin, ISequentialPostOptimizationMixin, INoMarginalProbabilityCalibrationMixin, \
    INoJointProbabilityCalibrationMixin
from mlrl.boosting.cython.learner cimport IAutomaticPartitionSamplingMixin, IAutomaticFeatureBinningMixin, \
    IAutomaticParallelRuleRefinementMixin, IAutomaticParallelStatisticUpdateMixin, IConstantShrinkageMixin, \
    INoL1RegularizationMixin, IL1RegularizationMixin, INoL2RegularizationMixin, IL2RegularizationMixin, \
    INoDefaultRuleMixin, IAutomaticDefaultRuleMixin, ICompleteHeadMixin, IDynamicPartialHeadMixin, \
    IFixedPartialHeadMixin, ISingleLabelHeadMixin, IAutomaticHeadMixin, IDenseStatisticsMixin, ISparseStatisticsMixin, \
    IAutomaticStatisticsMixin, IExampleWiseLogisticLossMixin, IExampleWiseSquaredErrorLossMixin, \
    IExampleWiseSquaredHingeLossMixin, ILabelWiseLogisticLossMixin, ILabelWiseSquaredErrorLossMixin, \
    ILabelWiseSquaredHingeLossMixin, INoLabelBinningMixin, IEqualWidthLabelBinningMixin, IAutomaticLabelBinningMixin, \
    IIsotonicMarginalProbabilityCalibrationMixin, IIsotonicJointProbabilityCalibrationMixin, \
    ILabelWiseBinaryPredictorMixin, IExampleWiseBinaryPredictorMixin, IGfmBinaryPredictorMixin, \
    IAutomaticBinaryPredictorMixin, ILabelWiseScorePredictorMixin, ILabelWiseProbabilityPredictorMixin, \
    IMarginalizedProbabilityPredictorMixin, IAutomaticProbabilityPredictorMixin, DdotFunction, DspmvFunction, \
    DsysvFunction

from libcpp.memory cimport unique_ptr


cdef extern from "boosting/learner_boomer.hpp" namespace "boosting" nogil:

    cdef cppclass IBoomerConfig"boosting::IBoomer::IConfig"(IAutomaticPartitionSamplingMixin,
                                                            IAutomaticFeatureBinningMixin,
                                                            IAutomaticParallelRuleRefinementMixin,
                                                            IAutomaticParallelStatisticUpdateMixin,
                                                            IConstantShrinkageMixin,
                                                            INoL1RegularizationMixin,
                                                            IL1RegularizationMixin,
                                                            INoL2RegularizationMixin,
                                                            IL2RegularizationMixin,
                                                            INoDefaultRuleMixin,
                                                            IAutomaticDefaultRuleMixin,
                                                            ICompleteHeadMixin,
                                                            IDynamicPartialHeadMixin,
                                                            IFixedPartialHeadMixin,
                                                            ISingleLabelHeadMixin,
                                                            IAutomaticHeadMixin,
                                                            IDenseStatisticsMixin,
                                                            ISparseStatisticsMixin,
                                                            IAutomaticStatisticsMixin,
                                                            IExampleWiseLogisticLossMixin,
                                                            IExampleWiseSquaredErrorLossMixin,
                                                            IExampleWiseSquaredHingeLossMixin,
                                                            ILabelWiseLogisticLossMixin,
                                                            ILabelWiseSquaredErrorLossMixin,
                                                            ILabelWiseSquaredHingeLossMixin,
                                                            INoLabelBinningMixin,
                                                            IEqualWidthLabelBinningMixin,
                                                            IAutomaticLabelBinningMixin,
                                                            IIsotonicMarginalProbabilityCalibrationMixin,
                                                            IIsotonicJointProbabilityCalibrationMixin,
                                                            ILabelWiseBinaryPredictorMixin,
                                                            IExampleWiseBinaryPredictorMixin,
                                                            IGfmBinaryPredictorMixin,
                                                            IAutomaticBinaryPredictorMixin,
                                                            ILabelWiseScorePredictorMixin,
                                                            ILabelWiseProbabilityPredictorMixin,
                                                            IMarginalizedProbabilityPredictorMixin,
                                                            IAutomaticProbabilityPredictorMixin,
                                                            ISequentialRuleModelAssemblageMixin,
                                                            IDefaultRuleMixin,
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
                                                            ISequentialPostOptimizationMixin,
                                                            INoMarginalProbabilityCalibrationMixin,
                                                            INoJointProbabilityCalibrationMixin):
        pass

    cdef cppclass IBoomer(IRuleLearner):
        pass


    unique_ptr[IBoomerConfig] createBoomerConfig()


    unique_ptr[IBoomer] createBoomer(unique_ptr[IBoomerConfig] configPtr, DdotFunction ddotFunction,
                                     DspmvFunction dspmvFunction, DsysvFunction dsysvFunction)


cdef class BoomerConfig(RuleLearnerConfig):

    # Attributes:

    cdef unique_ptr[IBoomerConfig] config_ptr


cdef class Boomer(RuleLearner):

    # Attributes:

    cdef unique_ptr[IBoomer] rule_learner_ptr
