from libcpp.memory cimport unique_ptr

from mlrl.common.cython.learner cimport IBeamSearchTopDownRuleInductionMixin, IDefaultRuleMixin, \
    IEqualFrequencyFeatureBinningMixin, IEqualWidthFeatureBinningMixin, \
    IExampleWiseStratifiedBiPartitionSamplingMixin, IExampleWiseStratifiedInstanceSamplingMixin, \
    IFeatureSamplingWithoutReplacementMixin, IGreedyTopDownRuleInductionMixin, \
    IInstanceSamplingWithoutReplacementMixin, IInstanceSamplingWithReplacementMixin, IIrepRulePruningMixin, \
    ILabelSamplingWithoutReplacementMixin, ILabelWiseStratifiedBiPartitionSamplingMixin, \
    ILabelWiseStratifiedInstanceSamplingMixin, INoFeatureBinningMixin, INoFeatureSamplingMixin, \
    INoInstanceSamplingMixin, INoLabelSamplingMixin, INoParallelPredictionMixin, INoParallelRuleRefinementMixin, \
    INoParallelStatisticUpdateMixin, INoPartitionSamplingMixin, INoRulePruningMixin, \
    INoSequentialPostOptimizationMixin, INoSizeStoppingCriterionMixin, INoTimeStoppingCriterionMixin, \
    IParallelPredictionMixin, IParallelRuleRefinementMixin, IParallelStatisticUpdateMixin, \
    IRandomBiPartitionSamplingMixin, IRoundRobinLabelSamplingMixin, IRuleLearner, ISequentialPostOptimizationMixin, \
    ISequentialRuleModelAssemblageMixin, ISizeStoppingCriterionMixin, ITimeStoppingCriterionMixin, RuleLearner, \
    RuleLearnerConfig

from mlrl.seco.cython.learner cimport IAccuracyHeuristicMixin, IAccuracyPruningHeuristicMixin, \
    ICoverageStoppingCriterionMixin, IFMeasureHeuristicMixin, IFMeasurePruningHeuristicMixin, IKlnLiftFunctionMixin, \
    ILabelWiseBinaryPredictorMixin, ILaplaceHeuristicMixin, ILaplacePruningHeuristicMixin, IMEstimateHeuristicMixin, \
    IMEstimatePruningHeuristicMixin, INoCoverageStoppingCriterionMixin, INoLiftFunctionMixin, IPartialHeadMixin, \
    IPeakLiftFunctionMixin, IPrecisionHeuristicMixin, IPrecisionPruningHeuristicMixin, IRecallHeuristicMixin, \
    IRecallPruningHeuristicMixin, ISingleLabelHeadMixin, IWraHeuristicMixin, IWraPruningHeuristicMixin


cdef extern from "mlrl/seco/learner_seco.hpp" namespace "seco" nogil:

    cdef cppclass IMultiLabelSeCoRuleLearnerConfig"seco::IMultiLabelSeCoRuleLearner::IConfig"(
            INoCoverageStoppingCriterionMixin,
            ICoverageStoppingCriterionMixin,
            ISingleLabelHeadMixin,
            IPartialHeadMixin,
            INoLiftFunctionMixin,
            IPeakLiftFunctionMixin,
            IKlnLiftFunctionMixin,
            IAccuracyHeuristicMixin,
            IAccuracyPruningHeuristicMixin,
            IFMeasureHeuristicMixin,
            IFMeasurePruningHeuristicMixin,
            IMEstimateHeuristicMixin,
            IMEstimatePruningHeuristicMixin,
            ILaplaceHeuristicMixin,
            ILaplacePruningHeuristicMixin,
            IPrecisionHeuristicMixin,
            IPrecisionPruningHeuristicMixin,
            IRecallHeuristicMixin,
            IRecallPruningHeuristicMixin,
            IWraHeuristicMixin,
            IWraPruningHeuristicMixin,
            ILabelWiseBinaryPredictorMixin,
            ISequentialRuleModelAssemblageMixin,
            IDefaultRuleMixin,
            IGreedyTopDownRuleInductionMixin,
            IBeamSearchTopDownRuleInductionMixin,
            INoFeatureBinningMixin,
            IEqualWidthFeatureBinningMixin,
            IEqualFrequencyFeatureBinningMixin,
            INoLabelSamplingMixin,
            IRoundRobinLabelSamplingMixin,
            ILabelSamplingWithoutReplacementMixin,
            INoInstanceSamplingMixin,
            IInstanceSamplingWithReplacementMixin,
            IInstanceSamplingWithoutReplacementMixin,
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
            INoSequentialPostOptimizationMixin,
            ISequentialPostOptimizationMixin):
        pass


    cdef cppclass IMultiLabelSeCoRuleLearner(IRuleLearner):
        pass


    unique_ptr[IMultiLabelSeCoRuleLearnerConfig] createMultiLabelSeCoRuleLearnerConfig()


    unique_ptr[IMultiLabelSeCoRuleLearner] createMultiLabelSeCoRuleLearner(
        unique_ptr[IMultiLabelSeCoRuleLearnerConfig] configPtr)


cdef class SeCoConfig(RuleLearnerConfig):

    # Attributes:

    cdef unique_ptr[IMultiLabelSeCoRuleLearnerConfig] config_ptr


cdef class SeCo(RuleLearner):

    # Attributes:

    cdef unique_ptr[IMultiLabelSeCoRuleLearner] rule_learner_ptr
