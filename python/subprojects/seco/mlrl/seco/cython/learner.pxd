from mlrl.common.cython.learner cimport IRuleLearner, RuleLearner, IRuleLearnerConfig, RuleLearnerConfig, \
    IBeamSearchTopDownMixin, IFeatureBinningMixin, ILabelSamplingMixin, IInstanceSamplingMixin, IFeatureSamplingMixin, \
    IPartitionSamplingMixin, IPruningMixin
from mlrl.seco.cython.heuristic cimport IFMeasureConfig, IMEstimateConfig
from mlrl.seco.cython.lift_function cimport IPeakLiftFunctionConfig, IKlnLiftFunctionConfig
from mlrl.seco.cython.stopping_criterion cimport ICoverageStoppingCriterionConfig

from libcpp.memory cimport unique_ptr


cdef extern from "seco/learner.hpp" namespace "seco" nogil:

    cdef cppclass ISeCoRuleLearnerConfig"seco::ISeCoRuleLearner::IConfig"(IRuleLearnerConfig,
                                                                          IBeamSearchTopDownMixin,
                                                                          IFeatureBinningMixin,
                                                                          ILabelSamplingMixin,
                                                                          IInstanceSamplingMixin,
                                                                          IFeatureSamplingMixin,
                                                                          IPartitionSamplingMixin,
                                                                          IPruningMixin):

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


    cdef cppclass ISeCoRuleLearner(IRuleLearner):
        pass


    unique_ptr[ISeCoRuleLearnerConfig] createSeCoRuleLearnerConfig()


    unique_ptr[ISeCoRuleLearner] createSeCoRuleLearner(unique_ptr[ISeCoRuleLearnerConfig] configPtr)


cdef class SeCoRuleLearnerConfig(RuleLearnerConfig):

    # Attributes:

    cdef unique_ptr[ISeCoRuleLearnerConfig] rule_learner_config_ptr


cdef class SeCoRuleLearner(RuleLearner):

    # Attributes:

    cdef unique_ptr[ISeCoRuleLearner] rule_learner_ptr
