from mlrl.seco.cython.heuristic cimport IFMeasureConfig, IMEstimateConfig
from mlrl.seco.cython.lift_function cimport IKlnLiftFunctionConfig, IPeakLiftFunctionConfig
from mlrl.seco.cython.stopping_criterion cimport ICoverageStoppingCriterionConfig


cdef extern from "mlrl/seco/learner.hpp" namespace "seco" nogil:

    cdef cppclass INoCoverageStoppingCriterionMixin:

        # Functions:

        void useNoCoverageStoppingCriterion()


    cdef cppclass ICoverageStoppingCriterionMixin:

        # Functions:

        ICoverageStoppingCriterionConfig& useCoverageStoppingCriterion()


    cdef cppclass ISingleOutputHeadMixin:

        # Functions:

        void useSingleOutputHeads()


    cdef cppclass IPartialHeadMixin:

        # Functions:

        void usePartialHeads()


    cdef cppclass INoLiftFunctionMixin:

        # Functions:

        void useNoLiftFunction()


    cdef cppclass IPeakLiftFunctionMixin:

        # Functions:

        IPeakLiftFunctionConfig& usePeakLiftFunction()


    cdef cppclass IKlnLiftFunctionMixin:

        # Functions:

        IKlnLiftFunctionConfig& useKlnLiftFunction()


    cdef cppclass IAccuracyHeuristicMixin:

        # Functions:

        void useAccuracyHeuristic()


    cdef cppclass IAccuracyPruningHeuristicMixin:

        # Functions:

        void useAccuracyPruningHeuristic()


    cdef cppclass IFMeasureHeuristicMixin:

        # Functions:

        IFMeasureConfig& useFMeasureHeuristic()


    cdef cppclass IFMeasurePruningHeuristicMixin:

        # Functions

        IFMeasureConfig& useFMeasurePruningHeuristic()


    cdef cppclass IMEstimateHeuristicMixin:

        # Functions:

        IMEstimateConfig& useMEstimateHeuristic()


    cdef cppclass IMEstimatePruningHeuristicMixin:

        # Functions:

        IMEstimateConfig& useMEstimatePruningHeuristic()


    cdef cppclass ILaplaceHeuristicMixin:

        # Functions:

        void useLaplaceHeuristic()


    cdef cppclass ILaplacePruningHeuristicMixin:

        # Functions:

        void useLaplacePruningHeuristic()


    cdef cppclass IPrecisionHeuristicMixin:

        # Functions:

        void usePrecisionHeuristic()


    cdef cppclass IPrecisionPruningHeuristicMixin:

        # Functions:

        void usePrecisionPruningHeuristic()


    cdef cppclass IRecallHeuristicMixin:

        # Functions:

        void useRecallHeuristic()


    cdef cppclass IRecallPruningHeuristicMixin:

        # Functions:

        void useRecallPruningHeuristic()


    cdef cppclass IWraHeuristicMixin:

        # Functions:

        void useWraHeuristic()


    cdef cppclass IWraPruningHeuristicMixin:

        # Functions:

        void useWraPruningHeuristic()


    cdef cppclass IOutputWiseBinaryPredictorMixin:

        # Functions:

        void useOutputWiseBinaryPredictor()
