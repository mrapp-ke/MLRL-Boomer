from boomer.common._arrays cimport uint32, intp, float64
from boomer.common._predictions cimport Prediction, PredictionCandidate, LabelWisePredictionCandidate
from boomer.common.input_data cimport LabelMatrix, AbstractRandomAccessLabelMatrix
from boomer.common.statistics cimport AbstractStatistics, AbstractStatisticsFactory, StatisticsFactory, \
    AbstractRefinementSearch, AbstractDecomposableRefinementSearch
from boomer.boosting.statistics cimport AbstractGradientStatistics
from boomer.boosting.label_wise_losses cimport AbstractLabelWiseLoss
from boomer.boosting.label_wise_rule_evaluation cimport AbstractLabelWiseRuleEvaluation

from libcpp cimport bool
from libcpp.memory cimport shared_ptr


cdef extern from "cpp/label_wise_statistics.h" namespace "boosting" nogil:

    cdef cppclass LabelWiseRefinementSearchImpl(AbstractDecomposableRefinementSearch):

        # Constructors:

        LabelWiseRefinementSearchImpl(shared_ptr[AbstractLabelWiseRuleEvaluation] ruleEvaluationPtr,
                                      intp numPredictions, const intp* labelIndices, intp numLabels,
                                      const float64* gradients, const float64* totalSumsOfGradients,
                                      const float64* hessians, const float64* totalSumsOfHessians) except +

        # Functions:

        void updateSearch(intp statisticIndex, uint32 weight)

        void resetSearch()

        LabelWisePredictionCandidate* calculateLabelWisePrediction(bool uncovered, bool accumulated) except +

        PredictionCandidate* calculateExampleWisePrediction(bool uncovered, bool accumulated) except +


    cdef cppclass LabelWiseStatisticsImpl(AbstractGradientStatistics):

        # Constructors:

        LabelWiseStatisticsImpl(shared_ptr[AbstractLabelWiseLoss] lossFunctionPtr,
                                shared_ptr[AbstractLabelWiseRuleEvaluation] ruleEvaluationPtr) except +

        # Functions:

        void resetSampledStatistics()

        void addSampledStatistic(intp statisticIndex, uint32 weight)

        void resetCoveredStatistics()

        void updateCoveredStatistic(intp statisticIndex, uint32 weight, bool remove)

        AbstractRefinementSearch* beginSearch(intp numLabelIndices, const intp* labelIndices)

        void applyPrediction(intp statisticIndex, const intp* labelIndices, Prediction* prediction)


    cdef cppclass LabelWiseStatisticsFactoryImpl(AbstractStatisticsFactory):

        # Constructors:

        LabelWiseStatisticsFactoryImpl(shared_ptr[AbstractLabelWiseLoss] lossFunctionPtr,
                                       shared_ptr[AbstractLabelWiseRuleEvaluation] ruleEvaluationPtr,
                                       shared_ptr[AbstractRandomAccessLabelMatrix] labelMatrixPtr) except +

        # Functions:

        AbstractStatistics* create()


cdef class LabelWiseStatisticsFactory(StatisticsFactory):

    # Attributes:

    cdef shared_ptr[AbstractLabelWiseLoss] loss_function_ptr

    cdef shared_ptr[AbstractLabelWiseRuleEvaluation] default_rule_evaluation_ptr

    cdef shared_ptr[AbstractLabelWiseRuleEvaluation] rule_evaluation_ptr

    # Functions:

    cdef AbstractStatistics* create_initial_statistics(self, LabelMatrix label_matrix)

    cdef AbstractStatistics* copy_statistics(self, AbstractStatistics* statistics)
