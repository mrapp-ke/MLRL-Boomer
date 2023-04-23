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


    cdef cppclass IPeakLiftFunctionMixin"seco::ISeCoRuleLearner::IPeakLiftFunctionMixin":

        # Functions:

        IPeakLiftFunctionConfig& usePeakLiftFunction()


    cdef cppclass IKlnLiftFunctionMixin"seco::ISeCoRuleLearner::IKlnLiftFunctionMixin":

        # Functions:

        IKlnLiftFunctionConfig& useKlnLiftFunction()


    cdef cppclass IAccuracyHeuristicMixin"seco::ISeCoRuleLearner::IAccuracyHeuristicMixin":

        # Functions:

        void useAccuracyHeuristic()

        void useAccuracyPruningHeuristic()


    cdef cppclass IFMeasureHeuristicMixin"seco::ISeCoRuleLearner::IFMeasureHeuristicMixin":

        # Functions:

        IFMeasureConfig& useFMeasureHeuristic()

        IFMeasureConfig& useFMeasurePruningHeuristic()


    cdef cppclass IMEstimateHeuristicMixin"seco::ISeCoRuleLearner::IMEstimateHeuristicMixin":

        # Functions:

        IMEstimateConfig& useMEstimateHeuristic()

        IMEstimateConfig& useMEstimatePruningHeuristic()


    cdef cppclass ILaplaceHeuristicMixin"seco::ISeCoRuleLearner::ILaplaceHeuristicMixin":

        # Functions:

        void useLaplaceHeuristic()

        void useLaplacePruningHeuristic()


    cdef cppclass IRecallHeuristicMixin"seco::ISeCoRuleLearner::IRecallHeuristicMixin":

        # Functions:

        void useRecallHeuristic()

        void useRecallPruningHeuristic()


    cdef cppclass IWraHeuristicMixin"seco::ISeCoRuleLearner::IWraHeuristicMixin":

        void useWraHeuristic()

        void useWraPruningHeuristic()


cdef class SeCoRuleLearnerConfig(RuleLearnerConfig):

    # Functions:

    cdef ISeCoRuleLearnerConfig* get_seco_rule_learner_config_ptr(self)
