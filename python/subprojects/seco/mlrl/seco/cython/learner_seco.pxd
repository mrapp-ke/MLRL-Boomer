from mlrl.common.cython.learner cimport IRuleLearner, RuleLearner, RuleLearnerConfig, \
    ISequentialRuleModelAssemblageMixin, IDefaultRuleMixin, IGreedyTopDownRuleInductionMixin, \
    IBeamSearchTopDownRuleInductionMixin, INoLabelSamplingMixin, ILabelSamplingWithoutReplacementMixin, \
    INoInstanceSamplingMixin, IInstanceSamplingWithoutReplacementMixin, IInstanceSamplingWithReplacementMixin, \
    ILabelWiseStratifiedInstanceSamplingMixin, IExampleWiseStratifiedInstanceSamplingMixin, INoFeatureSamplingMixin, \
    IFeatureSamplingWithoutReplacementMixin, INoPartitionSamplingMixin, IRandomBiPartitionSamplingMixin, \
    ILabelWiseStratifiedBiPartitionSamplingMixin, IExampleWiseStratifiedBiPartitionSamplingMixin, INoRulePruningMixin, \
    IIrepRulePruningMixin, INoParallelRuleRefinementMixin, IParallelRuleRefinementMixin, \
    INoParallelStatisticUpdateMixin, IParallelStatisticUpdateMixin, INoParallelPredictionMixin, \
    IParallelPredictionMixin, INoSizeStoppingCriterionMixin, ISizeStoppingCriterionMixin, \
    INoTimeStoppingCriterionMixin, ITimeStoppingCriterionMixin, INoSequentialPostOptimizationMixin, \
    ISequentialPostOptimizationMixin
from mlrl.seco.cython.learner cimport INoCoverageStoppingCriterionMixin, ICoverageStoppingCriterionMixin, \
    ISingleLabelHeadMixin, IPartialHeadMixin, INoLiftFunctionMixin, IPeakLiftFunctionMixin, IKlnLiftFunctionMixin, \
    IAccuracyHeuristicMixin, IAccuracyPruningHeuristicMixin, IFMeasureHeuristicMixin, IFMeasurePruningHeuristicMixin, \
    IMEstimateHeuristicMixin, IMEstimatePruningHeuristicMixin, ILaplaceHeuristicMixin, ILaplacePruningHeuristicMixin, \
    IPrecisionHeuristicMixin, IPrecisionPruningHeuristicMixin, IRecallHeuristicMixin, IRecallPruningHeuristicMixin, \
    IWraHeuristicMixin, IWraPruningHeuristicMixin, ILabelWiseBinaryPredictorMixin

from libcpp.memory cimport unique_ptr


cdef extern from "seco/learner_seco.hpp" namespace "seco" nogil:

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


cdef class MultiLabelSeCoRuleLearnerConfig(RuleLearnerConfig):

    # Attributes:

    cdef unique_ptr[IMultiLabelSeCoRuleLearnerConfig] rule_learner_config_ptr


cdef class MultiLabelSeCoRuleLearner(RuleLearner):

    # Attributes:

    cdef unique_ptr[IMultiLabelSeCoRuleLearner] rule_learner_ptr
