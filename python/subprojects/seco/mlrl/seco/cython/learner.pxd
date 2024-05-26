from mlrl.seco.cython.heuristic cimport IFMeasureConfig, IMEstimateConfig
from mlrl.seco.cython.lift_function cimport IKlnLiftFunctionConfig, IPeakLiftFunctionConfig
from mlrl.seco.cython.stopping_criterion cimport ICoverageStoppingCriterionConfig


cdef extern from "mlrl/seco/learner.hpp" namespace "seco" nogil:

    cdef cppclass INoCoverageStoppingCriterionMixin"seco::ISeCoRuleLearner::INoCoverageStoppingCriterionMixin":

        # Functions:

        void useNoCoverageStoppingCriterion()


    cdef cppclass ICoverageStoppingCriterionMixin"seco::ISeCoRuleLearner::ICoverageStoppingCriterionMixin":

        # Functions:

        ICoverageStoppingCriterionConfig& useCoverageStoppingCriterion()


    cdef cppclass ISingleOutputHeadMixin"seco::ISeCoRuleLearner::ISingleOutputHeadMixin":

        # Functions:

        void useSingleOutputHeads()


    cdef cppclass IPartialHeadMixin"seco::ISeCoRuleLearner::IPartialHeadMixin":

        # Functions:

        void usePartialHeads()


    cdef cppclass INoLiftFunctionMixin"seco::ISeCoRuleLearner::INoLiftFunctionMixin":

        # Functions:

        void useNoLiftFunction()


    cdef cppclass IPeakLiftFunctionMixin"seco::ISeCoRuleLearner::IPeakLiftFunctionMixin":

        # Functions:

        IPeakLiftFunctionConfig& usePeakLiftFunction()


    cdef cppclass IKlnLiftFunctionMixin"seco::ISeCoRuleLearner::IKlnLiftFunctionMixin":

        # Functions:

        IKlnLiftFunctionConfig& useKlnLiftFunction()


    cdef cppclass IAccuracyHeuristicMixin"seco::ISeCoRuleLearner::IAccuracyHeuristicMixin":

        # Functions:

        void useAccuracyHeuristic()


    cdef cppclass IAccuracyPruningHeuristicMixin"seco::ISeCoRuleLearner::IAccuracyPruningHeuristicMixin":

        # Functions:

        void useAccuracyPruningHeuristic()


    cdef cppclass IFMeasureHeuristicMixin"seco::ISeCoRuleLearner::IFMeasureHeuristicMixin":

        # Functions:

        IFMeasureConfig& useFMeasureHeuristic()


    cdef cppclass IFMeasurePruningHeuristicMixin"seco::ISeCoRuleLearner::IFMeasurePruningHeuristicMixin":

        # Functions

        IFMeasureConfig& useFMeasurePruningHeuristic()


    cdef cppclass IMEstimateHeuristicMixin"seco::ISeCoRuleLearner::IMEstimateHeuristicMixin":

        # Functions:

        IMEstimateConfig& useMEstimateHeuristic()


    cdef cppclass IMEstimatePruningHeuristicMixin"seco::ISeCoRuleLearner::IMEstimatePruningHeuristicMixin":

        # Functions:

        IMEstimateConfig& useMEstimatePruningHeuristic()


    cdef cppclass ILaplaceHeuristicMixin"seco::ISeCoRuleLearner::ILaplaceHeuristicMixin":

        # Functions:

        void useLaplaceHeuristic()


    cdef cppclass ILaplacePruningHeuristicMixin"seco::ISeCoRuleLearner::ILaplacePruningHeuristicMixin":

        # Functions:

        void useLaplacePruningHeuristic()


    cdef cppclass IPrecisionHeuristicMixin"seco::ISeCoRuleLearner::IPrecisionHeuristicMixin":

        # Functions:

        void usePrecisionHeuristic()


    cdef cppclass IPrecisionPruningHeuristicMixin"seco::ISeCoRuleLearner::IPrecisionPruningHeuristicMixin":

        # Functions:

        void usePrecisionPruningHeuristic()


    cdef cppclass IRecallHeuristicMixin"seco::ISeCoRuleLearner::IRecallHeuristicMixin":

        # Functions:

        void useRecallHeuristic()


    cdef cppclass IRecallPruningHeuristicMixin"seco::ISeCoRuleLearner::IRecallPruningHeuristicMixin":

        # Functions:

        void useRecallPruningHeuristic()


    cdef cppclass IWraHeuristicMixin"seco::ISeCoRuleLearner::IWraHeuristicMixin":

        # Functions:

        void useWraHeuristic()


    cdef cppclass IWraPruningHeuristicMixin"seco::ISeCoRuleLearner::IWraPruningHeuristicMixin":

        # Functions:

        void useWraPruningHeuristic()


    cdef cppclass ILabelWiseBinaryPredictorMixin"seco::ISeCoRuleLearner::ILabelWiseBinaryPredictorMixin":

        # Functions:

        void useLabelWiseBinaryPredictor()
