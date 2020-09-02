from boomer.common._arrays cimport uint32, intp, float64
from boomer.common._predictions cimport Prediction, PredictionCandidate, LabelWisePredictionCandidate
from boomer.common.input_data cimport LabelMatrix, AbstractRandomAccessLabelMatrix
from boomer.common.statistics cimport AbstractStatistics, AbstractStatisticsFactory, StatisticsFactory, \
    AbstractRefinementSearch
from boomer.boosting._lapack cimport Lapack
from boomer.boosting.statistics cimport AbstractGradientStatistics
from boomer.boosting.example_wise_losses cimport AbstractExampleWiseLoss
from boomer.boosting.example_wise_rule_evaluation cimport AbstractExampleWiseRuleEvaluation

from libcpp cimport bool
from libcpp.memory cimport shared_ptr


cdef extern from "cpp/example_wise_statistics.h" namespace "boosting" nogil:

    cdef cppclass ExampleWiseRefinementSearchImpl(AbstractRefinementSearch):

        # Constructors:

        ExampleWiseRefinementSearchImpl(shared_ptr[AbstractExampleWiseRuleEvaluation] ruleEvaluationPtr,
                                        shared_ptr[Lapack] lapackPtr, intp numPredictions, const intp* labelIndices,
                                        intp numLabels, const float64* gradients, const float64* totalSumsOfGradients,
                                        const float64* hessians, const float64* totalSumsOfHessians) except +

        # Functions:

        void updateSearch(intp statisticIndex, uint32 weight)

        void resetSearch()

        LabelWisePredictionCandidate* calculateLabelWisePrediction(bool uncovered, bool accumulated) except +

        PredictionCandidate* calculateExampleWisePrediction(bool uncovered, bool accumulated) except +


    cdef cppclass ExampleWiseStatisticsImpl(AbstractGradientStatistics):

        # Constructors:

        ExampleWiseStatisticsImpl(shared_ptr[AbstractExampleWiseLoss] lossFunctionPtr,
                                  shared_ptr[AbstractExampleWiseRuleEvaluation] ruleEvaluationPtr,
                                  shared_ptr[Lapack] lapackPtr) except +

        # Functions:

        void resetCoveredStatistics()

        void updateCoveredStatistic(intp statisticIndex, uint32 weight, bool remove)

        AbstractRefinementSearch* beginSearch(intp numLabelIndices, const intp* labelIndices)

        void applyPrediction(intp statisticIndex, const intp* labelIndices, Prediction* prediction)


    cdef cppclass ExampleWiseStatisticsFactoryImpl(AbstractStatisticsFactory):

        # Constructors:

        ExampleWiseStatisticsFactoryImpl(shared_ptr[AbstractExampleWiseLoss] lossFunctionPtr,
                                         shared_ptr[AbstractExampleWiseRuleEvaluation] ruleEvaluationPtr,
                                         shared_ptr[Lapack] lapackPtr,
                                         shared_ptr[AbstractRandomAccessLabelMatrix] labelMatrixPtr) except +

        # Functions:

        AbstractStatistics* create()


cdef class ExampleWiseStatisticsFactory(StatisticsFactory):

    # Attributes:

    cdef shared_ptr[AbstractExampleWiseLoss] loss_function_ptr

    cdef shared_ptr[AbstractExampleWiseRuleEvaluation] default_rule_evaluation_ptr

    cdef shared_ptr[AbstractExampleWiseRuleEvaluation] rule_evaluation_ptr

    cdef shared_ptr[Lapack] lapack_ptr

    # Functions:

    cdef AbstractStatistics* create_initial_statistics(self, LabelMatrix label_matrix)
