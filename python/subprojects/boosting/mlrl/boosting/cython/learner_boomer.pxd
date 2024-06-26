
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
    IParallelStatisticUpdateMixin, IPostPruningMixin, IPrePruningMixin, IRandomBiPartitionSamplingMixin, \
    IRoundRobinOutputSamplingMixin, ISequentialPostOptimizationMixin, ISequentialRuleModelAssemblageMixin, \
    ISizeStoppingCriterionMixin, ITimeStoppingCriterionMixin, RuleLearnerConfig
from mlrl.common.cython.learner_classification cimport ClassificationRuleLearner, IClassificationRuleLearner, \
    IExampleWiseStratifiedBiPartitionSamplingMixin, IExampleWiseStratifiedInstanceSamplingMixin, \
    IOutputWiseStratifiedBiPartitionSamplingMixin, IOutputWiseStratifiedInstanceSamplingMixin

from mlrl.boosting.cython.learner cimport DdotFunction, DspmvFunction, DsysvFunction, IAutomaticFeatureBinningMixin, \
    IAutomaticHeadMixin, IAutomaticParallelRuleRefinementMixin, IAutomaticParallelStatisticUpdateMixin, \
    ICompleteHeadMixin, IConstantShrinkageMixin, IDecomposableSquaredErrorLossMixin, IDynamicPartialHeadMixin, \
    IFixedPartialHeadMixin, IL1RegularizationMixin, IL2RegularizationMixin, INoL1RegularizationMixin, \
    INoL2RegularizationMixin, INonDecomposableSquaredErrorLossMixin, IOutputWiseScorePredictorMixin, \
    ISingleOutputHeadMixin
from mlrl.boosting.cython.learner_classification cimport IAutomaticBinaryPredictorMixin, IAutomaticDefaultRuleMixin, \
    IAutomaticLabelBinningMixin, IAutomaticPartitionSamplingMixin, IAutomaticProbabilityPredictorMixin, \
    IAutomaticStatisticsMixin, IDecomposableLogisticLossMixin, IDecomposableSquaredHingeLossMixin, \
    IDenseStatisticsMixin, IEqualWidthLabelBinningMixin, IExampleWiseBinaryPredictorMixin, IGfmBinaryPredictorMixin, \
    IIsotonicJointProbabilityCalibrationMixin, IIsotonicMarginalProbabilityCalibrationMixin, \
    IMarginalizedProbabilityPredictorMixin, INoDefaultRuleMixin, INoLabelBinningMixin, \
    INonDecomposableLogisticLossMixin, INonDecomposableSquaredHingeLossMixin, IOutputWiseBinaryPredictorMixin, \
    IOutputWiseProbabilityPredictorMixin, ISparseStatisticsMixin


cdef extern from "mlrl/boosting/learner_boomer.hpp" namespace "boosting" nogil:

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
                                                            IDefaultRuleMixin,
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

    cdef cppclass IBoomer(IClassificationRuleLearner):
        pass


    unique_ptr[IBoomerConfig] createBoomerConfig()


    unique_ptr[IBoomer] createBoomer(unique_ptr[IBoomerConfig] configPtr, DdotFunction ddotFunction,
                                     DspmvFunction dspmvFunction, DsysvFunction dsysvFunction)


cdef class BoomerConfig(RuleLearnerConfig):

    # Attributes:

    cdef unique_ptr[IBoomerConfig] config_ptr


cdef class Boomer(ClassificationRuleLearner):

    # Attributes:

    cdef unique_ptr[IBoomer] rule_learner_ptr
