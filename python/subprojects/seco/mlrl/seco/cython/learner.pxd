from mlrl.common.cython.learner cimport IRuleLearner, IRuleLearnerConfig, RuleLearner, RuleLearnerConfig
from mlrl.seco.cython.heuristic cimport IAccuracyConfig, IFMeasureConfig, ILaplaceConfig, IMEstimateConfig, \
    IPrecisionConfig, IRecallConfig, IWraConfig
from mlrl.seco.cython.lift_function cimport IPeakLiftFunctionConfig
from mlrl.seco.cython.predictor cimport ILabelWiseClassificationPredictorConfig
from mlrl.seco.cython.stopping_criterion cimport ICoverageStoppingCriterionConfig

from libcpp.memory cimport unique_ptr


cdef extern from "seco/learner.hpp" namespace "seco" nogil:

    cdef cppclass ISeCoRuleLearnerConfig"seco::ISeCoRuleLearner::IConfig"(IRuleLearnerConfig):

        # Functions:

        void useNoCoverageStoppingCriterion()

        ICoverageStoppingCriterionConfig& useCoverageStoppingCriterion()

        IAccuracyConfig& useAccuracyHeuristic()

        IFMeasureConfig& useFMeasureHeuristic()

        ILaplaceConfig& useLaplaceHeuristic()

        IMEstimateConfig& useMEstimateHeuristic()

        IPrecisionConfig& usePrecisionHeuristic()

        IRecallConfig& useRecallHeuristic()

        IWraConfig& useWraHeuristic()

        IAccuracyConfig& useAccuracyPruningHeuristic()

        IFMeasureConfig& useFMeasurePruningHeuristic()

        ILaplaceConfig& useLaplacePruningHeuristic()

        IMEstimateConfig& useMEstimatePruningHeuristic()

        IPrecisionConfig& usePrecisionPruningHeuristic()

        IRecallConfig& useRecallPruningHeuristic()

        IWraConfig& useWraPruningHeuristic()

        IPeakLiftFunctionConfig& usePeakLiftFunction()

        ILabelWiseClassificationPredictorConfig& useLabelWiseClassificationPredictor()


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
