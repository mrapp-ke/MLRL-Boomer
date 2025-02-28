
from libcpp.memory cimport unique_ptr

from mlrl.common.cython.learner cimport IBeamSearchTopDownRuleInductionMixin, IDefaultRuleMixin, \
    IEqualFrequencyFeatureBinningMixin, IEqualWidthFeatureBinningMixin, IFeatureSamplingWithoutReplacementMixin, \
    IGreedyTopDownRuleInductionMixin, IInstanceSamplingWithoutReplacementMixin, IInstanceSamplingWithReplacementMixin, \
    IIrepRulePruningMixin, INoFeatureBinningMixin, INoFeatureSamplingMixin, INoGlobalPruningMixin, \
    INoInstanceSamplingMixin, INoJointProbabilityCalibrationMixin, INoMarginalProbabilityCalibrationMixin, \
    INoOutputSamplingMixin, INoParallelPredictionMixin, INoParallelRuleRefinementMixin, \
    INoParallelStatisticUpdateMixin, INoPartitionSamplingMixin, INoPostProcessorMixin, INoRulePruningMixin, \
    INoSequentialPostOptimizationMixin, INoSizeStoppingCriterionMixin, INoTimeStoppingCriterionMixin, \
    IOutputSamplingWithoutReplacementMixin, IParallelPredictionMixin, IParallelRuleRefinementMixin, \
    IParallelStatisticUpdateMixin, IPostPruningMixin, IPrePruningMixin, IRandomBiPartitionSamplingMixin, IRNGMixin, \
    IRoundRobinOutputSamplingMixin, ISequentialPostOptimizationMixin, ISequentialRuleModelAssemblageMixin, \
    ISizeStoppingCriterionMixin, ITimeStoppingCriterionMixin, RuleLearnerConfig
from mlrl.common.cython.learner_classification cimport ClassificationRuleLearner, IClassificationRuleLearner, \
    IExampleWiseStratifiedBiPartitionSamplingMixin, IExampleWiseStratifiedInstanceSamplingMixin, \
    IOutputWiseStratifiedBiPartitionSamplingMixin, IOutputWiseStratifiedInstanceSamplingMixin
from mlrl.common.cython.learner_regression cimport IRegressionRuleLearner, RegressionRuleLearner

from mlrl.boosting.cython.learner cimport DdotFunction, DspmvFunction, DsysvFunction, IAutomaticFeatureBinningMixin, \
    IAutomaticHeadMixin, IAutomaticParallelRuleRefinementMixin, IAutomaticParallelStatisticUpdateMixin, \
    ICompleteHeadMixin, IConstantShrinkageMixin, IDecomposableSquaredErrorLossMixin, IDynamicPartialHeadMixin, \
    IFixedPartialHeadMixin, IFloat32StatisticsMixin, IFloat64StatisticsMixin, IL1RegularizationMixin, \
    IL2RegularizationMixin, INoL1RegularizationMixin, INoL2RegularizationMixin, INonDecomposableSquaredErrorLossMixin, \
    IOutputWiseScorePredictorMixin, ISingleOutputHeadMixin, SdotFunction, SspmvFunction, SsysvFunction
from mlrl.boosting.cython.learner_classification cimport IAutomaticBinaryPredictorMixin, IAutomaticDefaultRuleMixin, \
    IAutomaticLabelBinningMixin, IAutomaticPartitionSamplingMixin, IAutomaticProbabilityPredictorMixin, \
    IAutomaticStatisticsMixin, IDecomposableLogisticLossMixin, IDecomposableSquaredHingeLossMixin, \
    IDenseStatisticsMixin, IEqualWidthLabelBinningMixin, IExampleWiseBinaryPredictorMixin, IGfmBinaryPredictorMixin, \
    IIsotonicJointProbabilityCalibrationMixin, IIsotonicMarginalProbabilityCalibrationMixin, \
    IMarginalizedProbabilityPredictorMixin, INoDefaultRuleMixin, INoLabelBinningMixin, \
    INonDecomposableLogisticLossMixin, INonDecomposableSquaredHingeLossMixin, IOutputWiseBinaryPredictorMixin, \
    IOutputWiseProbabilityPredictorMixin, ISparseStatisticsMixin


cdef extern from "mlrl/boosting/learner_boomer_classifier.hpp" namespace "boosting" nogil:

    cdef cppclass IBoomerClassifierConfig"boosting::IBoomerClassifier::IConfig"(
        IRNGMixin,
        IAutomaticPartitionSamplingMixin,
        IAutomaticFeatureBinningMixin,
        IAutomaticParallelRuleRefinementMixin,
        IAutomaticParallelStatisticUpdateMixin,
        IConstantShrinkageMixin,
        IFloat32StatisticsMixin,
        IFloat64StatisticsMixin,
        INoL1RegularizationMixin,
        IL1RegularizationMixin,
        INoL2RegularizationMixin,
        IL2RegularizationMixin,
        INoDefaultRuleMixin,
        IDefaultRuleMixin,
        IAutomaticDefaultRuleMixin,
        ICompleteHeadMixin,
        IDynamicPartialHeadMixin,
        IFixedPartialHeadMixin,
        ISingleOutputHeadMixin,
        IAutomaticHeadMixin,
        IDenseStatisticsMixin,
        ISparseStatisticsMixin,
        IAutomaticStatisticsMixin,
        IDecomposableLogisticLossMixin,
        IDecomposableSquaredErrorLossMixin,
        IDecomposableSquaredHingeLossMixin,
        INonDecomposableLogisticLossMixin,
        INonDecomposableSquaredErrorLossMixin,
        INonDecomposableSquaredHingeLossMixin,
        INoLabelBinningMixin,
        IEqualWidthLabelBinningMixin,
        IAutomaticLabelBinningMixin,
        IIsotonicMarginalProbabilityCalibrationMixin,
        IIsotonicJointProbabilityCalibrationMixin,
        IOutputWiseBinaryPredictorMixin,
        IExampleWiseBinaryPredictorMixin,
        IGfmBinaryPredictorMixin,
        IAutomaticBinaryPredictorMixin,
        IOutputWiseScorePredictorMixin,
        IOutputWiseProbabilityPredictorMixin,
        IMarginalizedProbabilityPredictorMixin,
        IAutomaticProbabilityPredictorMixin,
        ISequentialRuleModelAssemblageMixin,
        IGreedyTopDownRuleInductionMixin,
        IBeamSearchTopDownRuleInductionMixin,
        INoPostProcessorMixin,
        INoFeatureBinningMixin,
        IEqualWidthFeatureBinningMixin,
        IEqualFrequencyFeatureBinningMixin,
        INoOutputSamplingMixin,
        IRoundRobinOutputSamplingMixin,
        IOutputSamplingWithoutReplacementMixin,
        INoInstanceSamplingMixin,
        IInstanceSamplingWithoutReplacementMixin,
        IInstanceSamplingWithReplacementMixin,
        IOutputWiseStratifiedInstanceSamplingMixin,
        IExampleWiseStratifiedInstanceSamplingMixin,
        INoFeatureSamplingMixin,
        IFeatureSamplingWithoutReplacementMixin,
        INoPartitionSamplingMixin,
        IRandomBiPartitionSamplingMixin,
        IOutputWiseStratifiedBiPartitionSamplingMixin,
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

    cdef cppclass IBoomerClassifier(IClassificationRuleLearner):
        pass


    unique_ptr[IBoomerClassifierConfig] createBoomerClassifierConfig()


    unique_ptr[IBoomerClassifier] createBoomerClassifier(unique_ptr[IBoomerClassifierConfig] configPtr,
                                                         SdotFunction sdotFunction, DdotFunction ddotFunction,
                                                         SspmvFunction sspmvFunction, DspmvFunction dspmvFunction,
                                                         SsysvFunction ssysvFunction, DsysvFunction dsysvFunction)


cdef extern from "mlrl/boosting/learner_boomer_regressor.hpp" namespace "boosting" nogil:

    cdef cppclass IBoomerRegressorConfig"boosting::IBoomerRegressor::IConfig"(
        IAutomaticPartitionSamplingMixin,
        IAutomaticFeatureBinningMixin,
        IAutomaticParallelRuleRefinementMixin,
        IAutomaticParallelStatisticUpdateMixin,
        INoPostProcessorMixin,
        IConstantShrinkageMixin,
        IFloat32StatisticsMixin,
        IFloat64StatisticsMixin,
        INoL1RegularizationMixin,
        IL1RegularizationMixin,
        INoL2RegularizationMixin,
        IL2RegularizationMixin,
        INoDefaultRuleMixin,
        IDefaultRuleMixin,
        IAutomaticDefaultRuleMixin,
        ICompleteHeadMixin,
        IDynamicPartialHeadMixin,
        IFixedPartialHeadMixin,
        ISingleOutputHeadMixin,
        IAutomaticHeadMixin,
        IDenseStatisticsMixin,
        ISparseStatisticsMixin,
        IAutomaticStatisticsMixin,
        IDecomposableSquaredErrorLossMixin,
        INonDecomposableSquaredErrorLossMixin,
        IOutputWiseScorePredictorMixin,
        ISequentialRuleModelAssemblageMixin,
        IGreedyTopDownRuleInductionMixin,
        IBeamSearchTopDownRuleInductionMixin,
        INoFeatureBinningMixin,
        IEqualWidthFeatureBinningMixin,
        IEqualFrequencyFeatureBinningMixin,
        INoOutputSamplingMixin,
        IRoundRobinOutputSamplingMixin,
        IOutputSamplingWithoutReplacementMixin,
        INoInstanceSamplingMixin,
        IInstanceSamplingWithoutReplacementMixin,
        IInstanceSamplingWithReplacementMixin,
        INoFeatureSamplingMixin,
        IFeatureSamplingWithoutReplacementMixin,
        INoPartitionSamplingMixin,
        IRandomBiPartitionSamplingMixin,
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
        pass

    cdef cppclass IBoomerRegressor(IRegressionRuleLearner):
        pass


    unique_ptr[IBoomerRegressorConfig] createBoomerRegressorConfig()


    unique_ptr[IBoomerRegressor] createBoomerRegressor(unique_ptr[IBoomerRegressorConfig] configPtr,
                                                       SdotFunction sdotFunction, DdotFunction ddotFunction,
                                                       SspmvFunction sspmvFunction, DspmvFunction dspmvFunction,
                                                       SsysvFunction ssysvFunction, DsysvFunction dsysvFunction)


cdef class BoomerClassifierConfig(RuleLearnerConfig):

    # Attributes:

    cdef unique_ptr[IBoomerClassifierConfig] config_ptr


cdef class BoomerClassifier(ClassificationRuleLearner):

    # Attributes:

    cdef unique_ptr[IBoomerClassifier] rule_learner_ptr


cdef class BoomerRegressorConfig(RuleLearnerConfig):

    # Attributes:

    cdef unique_ptr[IBoomerRegressorConfig] config_ptr


cdef class BoomerRegressor(RegressionRuleLearner):

    # Attributes:

    cdef unique_ptr[IBoomerRegressor] rule_learner_ptr
