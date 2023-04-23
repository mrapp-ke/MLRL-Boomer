from mlrl.common.cython.learner cimport IRuleLearner, RuleLearner, IDefaultRuleMixin, IBeamSearchTopDownMixin, \
    INoFeatureBinningMixin, INoLabelSamplingMixin, ILabelSamplingWithoutReplacementMixin, INoInstanceSamplingMixin, \
    IInstanceSamplingWithoutReplacementMixin, IInstanceSamplingWithReplacementMixin, \
    ILabelWiseStratifiedInstanceSamplingMixin, IExampleWiseStratifiedInstanceSamplingMixin, INoFeatureSamplingMixin, \
    IFeatureSamplingWithoutReplacementMixin, INoPartitionSamplingMixin, IRandomBiPartitionSamplingMixin, \
    ILabelWiseStratifiedBiPartitionSamplingMixin, IExampleWiseStratifiedBiPartitionSamplingMixin, INoRulePruningMixin, \
    IIrepRulePruningMixin, INoParallelRuleRefinementMixin, IParallelRuleRefinementMixin, \
    INoParallelStatisticUpdateMixin, IParallelStatisticUpdateMixin, INoParallelPredictionMixin, \
    IParallelPredictionMixin, INoSizeStoppingCriterionMixin, ISizeStoppingCriterionMixin, ITimeStoppingCriterionMixin, \
    INoGlobalPruningMixin, ISequentialPostOptimizationMixin
from mlrl.seco.cython.learner cimport ISeCoRuleLearnerConfig, SeCoRuleLearnerConfig, ICoverageStoppingCriterionMixin, \
    IPartialHeadMixin, IPeakLiftFunctionMixin, IKlnLiftFunctionMixin, IAccuracyHeuristicMixin, \
    IAccuracyPruningHeuristicMixin, IFMeasureHeuristicMixin, IFMeasurePruningHeuristicMixin, IMEstimateHeuristicMixin, \
    IMEstimatePruningHeuristicMixin, ILaplaceHeuristicMixin, ILaplacePruningHeuristicMixin, IRecallHeuristicMixin, \
    IRecallPruningHeuristicMixin, IWraHeuristicMixin, IWraPruningHeuristicMixin

from libcpp.memory cimport unique_ptr


cdef extern from "seco/learner_seco.hpp" namespace "seco" nogil:

    cdef cppclass IMultiLabelSeCoRuleLearnerConfig"seco::IMultiLabelSeCoRuleLearner::IConfig"(
            ISeCoRuleLearnerConfig,
            ICoverageStoppingCriterionMixin,
            IPartialHeadMixin,
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
            IRecallHeuristicMixin,
            IRecallPruningHeuristicMixin,
            IWraHeuristicMixin,
            IWraPruningHeuristicMixin,
            IDefaultRuleMixin,
            IBeamSearchTopDownMixin,
            INoFeatureBinningMixin,
            INoLabelSamplingMixin,
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
            ITimeStoppingCriterionMixin,
            INoGlobalPruningMixin,
            ISequentialPostOptimizationMixin):
        pass


    cdef cppclass IMultiLabelSeCoRuleLearner(IRuleLearner):
        pass


    unique_ptr[IMultiLabelSeCoRuleLearnerConfig] createMultiLabelSeCoRuleLearnerConfig()


    unique_ptr[IMultiLabelSeCoRuleLearner] createMultiLabelSeCoRuleLearner(
        unique_ptr[IMultiLabelSeCoRuleLearnerConfig] configPtr)


cdef class MultiLabelSeCoRuleLearnerConfig(SeCoRuleLearnerConfig):

    # Attributes:

    cdef unique_ptr[IMultiLabelSeCoRuleLearnerConfig] rule_learner_config_ptr


cdef class MultiLabelSeCoRuleLearner(RuleLearner):

    # Attributes:

    cdef unique_ptr[IMultiLabelSeCoRuleLearner] rule_learner_ptr
