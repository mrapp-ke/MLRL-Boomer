from libcpp.memory cimport unique_ptr

from mlrl.common.cython.learner cimport IBeamSearchTopDownRuleInductionMixin, IDefaultRuleMixin, \
    IEqualFrequencyFeatureBinningMixin, IEqualWidthFeatureBinningMixin, IFeatureSamplingWithoutReplacementMixin, \
    IGreedyTopDownRuleInductionMixin, IInstanceSamplingWithoutReplacementMixin, IInstanceSamplingWithReplacementMixin, \
    IIrepRulePruningMixin, INoFeatureBinningMixin, INoFeatureSamplingMixin, INoInstanceSamplingMixin, \
    INoOutputSamplingMixin, INoParallelPredictionMixin, INoParallelRuleRefinementMixin, \
    INoParallelStatisticUpdateMixin, INoPartitionSamplingMixin, INoRulePruningMixin, \
    INoSequentialPostOptimizationMixin, INoSizeStoppingCriterionMixin, INoTimeStoppingCriterionMixin, \
    IOutputSamplingWithoutReplacementMixin, IParallelPredictionMixin, IParallelRuleRefinementMixin, \
    IParallelStatisticUpdateMixin, IRandomBiPartitionSamplingMixin, IRoundRobinOutputSamplingMixin, IRuleLearner, \
    ISequentialPostOptimizationMixin, ISequentialRuleModelAssemblageMixin, ISizeStoppingCriterionMixin, \
    ITimeStoppingCriterionMixin, RuleLearner, RuleLearnerConfig
from mlrl.common.cython.learner_classification cimport IExampleWiseStratifiedBiPartitionSamplingMixin, \
    IExampleWiseStratifiedInstanceSamplingMixin, IOutputWiseStratifiedBiPartitionSamplingMixin, \
    IOutputWiseStratifiedInstanceSamplingMixin

from mlrl.seco.cython.learner cimport IAccuracyHeuristicMixin, IAccuracyPruningHeuristicMixin, \
    ICoverageStoppingCriterionMixin, IFMeasureHeuristicMixin, IFMeasurePruningHeuristicMixin, IKlnLiftFunctionMixin, \
    ILaplaceHeuristicMixin, ILaplacePruningHeuristicMixin, IMEstimateHeuristicMixin, IMEstimatePruningHeuristicMixin, \
    INoCoverageStoppingCriterionMixin, INoLiftFunctionMixin, IOutputWiseBinaryPredictorMixin, IPartialHeadMixin, \
    IPeakLiftFunctionMixin, IPrecisionHeuristicMixin, IPrecisionPruningHeuristicMixin, IRecallHeuristicMixin, \
    IRecallPruningHeuristicMixin, ISingleOutputHeadMixin, IWraHeuristicMixin, IWraPruningHeuristicMixin


cdef extern from "mlrl/seco/learner_seco.hpp" namespace "seco" nogil:

    cdef cppclass IMultiLabelSeCoRuleLearnerConfig"seco::IMultiLabelSeCoRuleLearner::IConfig"(
            INoCoverageStoppingCriterionMixin,
            ICoverageStoppingCriterionMixin,
            ISingleOutputHeadMixin,
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
            IOutputWiseBinaryPredictorMixin,
            ISequentialRuleModelAssemblageMixin,
            IDefaultRuleMixin,
            IGreedyTopDownRuleInductionMixin,
            IBeamSearchTopDownRuleInductionMixin,
            INoFeatureBinningMixin,
            IEqualWidthFeatureBinningMixin,
            IEqualFrequencyFeatureBinningMixin,
            INoOutputSamplingMixin,
            IRoundRobinOutputSamplingMixin,
            IOutputSamplingWithoutReplacementMixin,
            INoInstanceSamplingMixin,
            IInstanceSamplingWithReplacementMixin,
            IInstanceSamplingWithoutReplacementMixin,
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
