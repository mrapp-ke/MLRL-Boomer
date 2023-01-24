from mlrl.common.cython.learner cimport IRuleLearnerConfig, RuleLearnerConfig
from mlrl.seco.cython.heuristic cimport IFMeasureConfig, IMEstimateConfig
from mlrl.seco.cython.lift_function cimport IPeakLiftFunctionConfig, IKlnLiftFunctionConfig
from mlrl.seco.cython.stopping_criterion cimport ICoverageStoppingCriterionConfig


cdef extern from "seco/learner.hpp" namespace "seco" nogil:

    cdef cppclass ISeCoRuleLearnerConfig"seco::ISeCoRuleLearner::IConfig"(IRuleLearnerConfig):

        # Functions:

        void useNoCoverageStoppingCriterion()

        void useSingleLabelHeads()

        void useNoLiftFunction()

        void usePrecisionHeuristic()

        void usePrecisionPruningHeuristic()

        void useLabelWiseBinaryPredictor()


    cdef cppclass ICoverageStoppingCriterionMixin"seco::ISeCoRuleLearner::ICoverageStoppingCriterionMixin":

        # Functions:

        ICoverageStoppingCriterionConfig& useCoverageStoppingCriterion()


    cdef cppclass IPartialHeadMixin"seco::ISeCoRuleLearner::IPartialHeadMixin":

        # Functions:

        void usePartialHeads()

        IPeakLiftFunctionConfig& usePeakLiftFunction()

        IKlnLiftFunctionConfig& useKlnLiftFunction()


    cdef cppclass IAccuracyMixin"seco::ISeCoRuleLearner::IAccuracyMixin":

        # Functions:

        void useAccuracyHeuristic()

        void useAccuracyPruningHeuristic()


    cdef cppclass IFMeasureMixin"seco::ISeCoRuleLearner::IFMeasureMixin":

        # Functions:

        IFMeasureConfig& useFMeasureHeuristic()

        IFMeasureConfig& useFMeasurePruningHeuristic()


    cdef cppclass IMEstimateMixin"seco::ISeCoRuleLearner::IMEstimateMixin":

        # Functions:

        IMEstimateConfig& useMEstimateHeuristic()

        IMEstimateConfig& useMEstimatePruningHeuristic()


    cdef cppclass ILaplaceMixin"seco::ISeCoRuleLearner::ILaplaceMixin":

        # Functions:

        void useLaplaceHeuristic()

        void useLaplacePruningHeuristic()


    cdef cppclass IRecallMixin"seco::ISeCoRuleLearner::IRecallMixin":

        # Functions:

        void useRecallHeuristic()

        void useRecallPruningHeuristic()


    cdef cppclass IWraMixin"seco::ISeCoRuleLearner::IWraMixin":

        void useWraHeuristic()

        void useWraPruningHeuristic()


cdef class SeCoRuleLearnerConfig(RuleLearnerConfig):

    # Functions:

    cdef ISeCoRuleLearnerConfig* get_seco_rule_learner_config_ptr(self)
