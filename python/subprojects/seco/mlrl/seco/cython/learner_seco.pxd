from mlrl.common.cython.learner cimport IRuleLearner, RuleLearner, IRuleLearnerConfig, RuleLearnerConfig, \
    IBeamSearchTopDownMixin, IFeatureBinningMixin, ILabelSamplingMixin, IInstanceSamplingMixin, IFeatureSamplingMixin, \
    IPartitionSamplingMixin, IPruningMixin, IMultiThreadingMixin, ISizeStoppingCriterionMixin, \
    ITimeStoppingCriterionMixin, IMeasureStoppingCriterionMixin
from mlrl.seco.cython.heuristic cimport IFMeasureConfig, IMEstimateConfig
from mlrl.seco.cython.lift_function cimport IPeakLiftFunctionConfig, IKlnLiftFunctionConfig
from mlrl.seco.cython.stopping_criterion cimport ICoverageStoppingCriterionConfig

from libcpp.memory cimport unique_ptr


cdef extern from "seco/learner_seco.hpp" namespace "seco" nogil:

    cdef cppclass IMultiLabelSeCoRuleLearnerConfig"seco::IMultiLabelSeCoRuleLearner::IConfig"(IRuleLearnerConfig,
                                                                                              IBeamSearchTopDownMixin,
                                                                                              IFeatureBinningMixin,
                                                                                              ILabelSamplingMixin,
                                                                                              IInstanceSamplingMixin,
                                                                                              IFeatureSamplingMixin,
                                                                                              IPartitionSamplingMixin,
                                                                                              IPruningMixin,
                                                                                              IMultiThreadingMixin,
                                                                                              ISizeStoppingCriterionMixin,
                                                                                              ITimeStoppingCriterionMixin,
                                                                                              IMeasureStoppingCriterionMixin):

        # Functions:

        void useNoCoverageStoppingCriterion()

        ICoverageStoppingCriterionConfig& useCoverageStoppingCriterion()

        void useSingleLabelHeads()

        void usePartialHeads()

        void useAccuracyHeuristic()

        IFMeasureConfig& useFMeasureHeuristic()

        void useLaplaceHeuristic()

        IMEstimateConfig& useMEstimateHeuristic()

        void usePrecisionHeuristic()

        void useRecallHeuristic()

        void useWraHeuristic()

        void useAccuracyPruningHeuristic()

        IFMeasureConfig& useFMeasurePruningHeuristic()

        void useLaplacePruningHeuristic()

        IMEstimateConfig& useMEstimatePruningHeuristic()

        void usePrecisionPruningHeuristic()

        void useRecallPruningHeuristic()

        void useWraPruningHeuristic()

        IPeakLiftFunctionConfig& usePeakLiftFunction()

        IKlnLiftFunctionConfig& useKlnLiftFunction()

        void useLabelWiseClassificationPredictor()


    cdef cppclass IMultiLabelSeCoRuleLearner(IRuleLearner):
        pass


    unique_ptr[IMultiLabelSeCoRuleLearnerConfig] createMultiLabelSeCoRuleLearnerConfig()


    unique_ptr[IMultiLabelSeCoRuleLearner] createMultiLabelSeCoRuleLearner(unique_ptr[IMultiLabelSeCoRuleLearnerConfig] configPtr)


cdef class MultiLabelSeCoRuleLearnerConfig(RuleLearnerConfig):

    # Attributes:

    cdef unique_ptr[IMultiLabelSeCoRuleLearnerConfig] rule_learner_config_ptr


cdef class MultiLabelSeCoRuleLearner(RuleLearner):

    # Attributes:

    cdef unique_ptr[IMultiLabelSeCoRuleLearner] rule_learner_ptr
