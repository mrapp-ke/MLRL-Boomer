from mlrl.common.cython.learner cimport IRuleLearner, RuleLearner, IBeamSearchTopDownMixin, ILabelSamplingMixin, \
    IInstanceSamplingMixin, IFeatureSamplingMixin, IPartitionSamplingMixin, IRulePruningMixin, IMultiThreadingMixin, \
    ISizeStoppingCriterionMixin, ITimeStoppingCriterionMixin, ISequentialPostOptimizationMixin
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
            IBeamSearchTopDownMixin,
            ILabelSamplingMixin,
            IInstanceSamplingMixin,
            IFeatureSamplingMixin,
            IPartitionSamplingMixin,
            IRulePruningMixin,
            IMultiThreadingMixin,
            ISizeStoppingCriterionMixin,
            ITimeStoppingCriterionMixin,
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
