from libcpp.memory cimport unique_ptr

from mlrl.common.cython.learner cimport IBeamSearchTopDownRuleInductionMixin, IDefaultRuleMixin, \
    IEqualFrequencyFeatureBinningMixin, IEqualWidthFeatureBinningMixin, IFeatureSamplingWithoutReplacementMixin, \
    IGreedyTopDownRuleInductionMixin, IInstanceSamplingWithoutReplacementMixin, IInstanceSamplingWithReplacementMixin, \
    IIrepRulePruningMixin, INoFeatureBinningMixin, INoFeatureSamplingMixin, INoInstanceSamplingMixin, \
    INoOutputSamplingMixin, INoParallelPredictionMixin, INoParallelRuleRefinementMixin, \
    INoParallelStatisticUpdateMixin, INoPartitionSamplingMixin, INoRulePruningMixin, \
    INoSequentialPostOptimizationMixin, INoSizeStoppingCriterionMixin, INoTimeStoppingCriterionMixin, \
    IOutputSamplingWithoutReplacementMixin, IParallelPredictionMixin, IParallelRuleRefinementMixin, \
    IParallelStatisticUpdateMixin, IRandomBiPartitionSamplingMixin, IRNGMixin, IRoundRobinOutputSamplingMixin, \
    ISequentialPostOptimizationMixin, ISequentialRuleModelAssemblageMixin, ISizeStoppingCriterionMixin, \
    ITimeStoppingCriterionMixin, RuleLearnerConfig
from mlrl.common.cython.learner_classification cimport ClassificationRuleLearner, IClassificationRuleLearner, \
    IExampleWiseStratifiedBiPartitionSamplingMixin, IExampleWiseStratifiedInstanceSamplingMixin, \
    IOutputWiseStratifiedBiPartitionSamplingMixin, IOutputWiseStratifiedInstanceSamplingMixin

from mlrl.seco.cython.learner cimport IAccuracyHeuristicMixin, IAccuracyPruningHeuristicMixin, \
    ICoverageStoppingCriterionMixin, IFMeasureHeuristicMixin, IFMeasurePruningHeuristicMixin, IKlnLiftFunctionMixin, \
    ILaplaceHeuristicMixin, ILaplacePruningHeuristicMixin, IMEstimateHeuristicMixin, IMEstimatePruningHeuristicMixin, \
    INoCoverageStoppingCriterionMixin, INoLiftFunctionMixin, IOutputWiseBinaryPredictorMixin, IPartialHeadMixin, \
    IPeakLiftFunctionMixin, IPrecisionHeuristicMixin, IPrecisionPruningHeuristicMixin, IRecallHeuristicMixin, \
    IRecallPruningHeuristicMixin, ISingleOutputHeadMixin, IWraHeuristicMixin, IWraPruningHeuristicMixin


cdef extern from "mlrl/seco/learner_seco_classifier.hpp" namespace "seco" nogil:

    cdef cppclass ISeCoClassifierConfig"seco::ISeCoClassifier::IConfig"(IRNGMixin,
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


    cdef cppclass ISeCoClassifier(IClassificationRuleLearner):
        pass


    unique_ptr[ISeCoClassifierConfig] createSeCoClassifierConfig()


    unique_ptr[ISeCoClassifier] createSeCoClassifier(unique_ptr[ISeCoClassifierConfig] configPtr)


cdef class SeCoClassifierConfig(RuleLearnerConfig):

    # Attributes:

    cdef unique_ptr[ISeCoClassifierConfig] config_ptr


cdef class SeCoClassifier(ClassificationRuleLearner):

    # Attributes:

    cdef unique_ptr[ISeCoClassifier] rule_learner_ptr
