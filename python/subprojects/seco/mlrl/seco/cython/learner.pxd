from mlrl.common.cython.learner cimport IRuleLearnerConfig, RuleLearnerConfig
from mlrl.seco.cython.heuristic cimport IFMeasureConfig, IMEstimateConfig
from mlrl.seco.cython.lift_function cimport IPeakLiftFunctionConfig, IKlnLiftFunctionConfig
from mlrl.seco.cython.stopping_criterion cimport ICoverageStoppingCriterionConfig


cdef extern from "seco/learner.hpp" namespace "seco" nogil:

    cdef cppclass ISeCoRuleLearnerConfig"seco::ISeCoRuleLearner::IConfig"(IRuleLearnerConfig):

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


cdef class SeCoRuleLearnerConfig(RuleLearnerConfig):

    # Functions:

    cdef ISeCoRuleLearnerConfig* get_seco_rule_learner_config_ptr(self)
