from mlrl.common.cython.learner cimport IRuleLearner, IRuleLearnerConfig, RuleLearner, RuleLearnerConfig
from mlrl.seco.cython.heuristic cimport AccuracyConfigImpl, FMeasureConfigImpl, LaplaceConfigImpl, \
    MEstimateConfigImpl, PrecisionConfigImpl, RecallConfigImpl, WraConfigImpl
from mlrl.seco.cython.lift_function cimport PeakLiftFunctionConfigImpl
from mlrl.seco.cython.predictor cimport LabelWiseClassificationPredictorConfigImpl
from mlrl.seco.cython.stopping_criterion cimport ICoverageStoppingCriterionConfig

from libcpp.memory cimport unique_ptr


cdef extern from "seco/learner.hpp" namespace "seco" nogil:

    cdef cppclass ISeCoRuleLearnerConfig"seco::ISeCoRuleLearner::IConfig"(IRuleLearnerConfig):

        # Functions:

        void useNoCoverageStoppingCriterion()

        ICoverageStoppingCriterionConfig& useCoverageStoppingCriterion()

        AccuracyConfigImpl& useAccuracyHeuristic()

        FMeasureConfigImpl& useFMeasureHeuristic()

        LaplaceConfigImpl& useLaplaceHeuristic()

        MEstimateConfigImpl& useMEstimateHeuristic()

        PrecisionConfigImpl& usePrecisionHeuristic()

        RecallConfigImpl& useRecallHeuristic()

        WraConfigImpl& useWraHeuristic()

        AccuracyConfigImpl& useAccuracyPruningHeuristic()

        FMeasureConfigImpl& useFMeasurePruningHeuristic()

        LaplaceConfigImpl& useLaplacePruningHeuristic()

        MEstimateConfigImpl& useMEstimatePruningHeuristic()

        PrecisionConfigImpl& usePrecisionPruningHeuristic()

        RecallConfigImpl& useRecallPruningHeuristic()

        WraConfigImpl& useWraPruningHeuristic()

        PeakLiftFunctionConfigImpl& usePeakLiftFunction()

        LabelWiseClassificationPredictorConfigImpl& useLabelWiseClassificationPredictor()


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
