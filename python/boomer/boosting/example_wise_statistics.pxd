from boomer.common._arrays cimport uint32, intp, float64
from boomer.boosting._lapack cimport Lapack
from boomer.common.input_data cimport AbstractRandomAccessLabelMatrix
from boomer.common.statistics cimport AbstractStatistics, AbstractRefinementSearch
from boomer.common.head_refinement cimport HeadCandidate
from boomer.common.rule_evaluation cimport DefaultPrediction, Prediction, LabelWisePrediction
from boomer.boosting.statistics cimport GradientStatistics, AbstractGradientStatistics
from boomer.boosting.example_wise_losses cimport AbstractExampleWiseLoss
from boomer.boosting.example_wise_rule_evaluation cimport ExampleWiseRuleEvaluationImpl

from libcpp cimport bool
from libcpp.memory cimport shared_ptr


cdef extern from "cpp/example_wise_statistics.h" namespace "boosting" nogil:

    cdef cppclass ExampleWiseRefinementSearchImpl(AbstractRefinementSearch):

        # Constructors:

        ExampleWiseRefinementSearchImpl(shared_ptr[ExampleWiseRuleEvaluationImpl] ruleEvaluationPtr,
                                        shared_ptr[Lapack] lapackPtr, intp numPredictions, const intp* labelIndices,
                                        intp numLabels, const float64* gradients, const float64* totalSumsOfGradients,
                                        const float64* hessians, const float64* totalSumsOfHessians) except +

        # Functions:

        void updateSearch(intp statisticIndex, uint32 weight)

        void resetSearch()

        LabelWisePrediction* calculateLabelWisePrediction(bool uncovered, bool accumulated) except +

        Prediction* calculateExampleWisePrediction(bool uncovered, bool accumulated) except +


    cdef cppclass ExampleWiseStatisticsImpl(AbstractGradientStatistics):

        # Constructors:

        ExampleWiseStatisticsImpl(shared_ptr[AbstractExampleWiseLoss] lossFunctionPtr,
                                  shared_ptr[ExampleWiseRuleEvaluationImpl] ruleEvaluationPtr,
                                  shared_ptr[Lapack] lapackPtr) except +

        # Functions:

        void applyDefaultPrediction(shared_ptr[AbstractRandomAccessLabelMatrix] labelMatrixPtr,
                                    DefaultPrediction* defaultPrediction)

        void resetCoveredStatistics()

        void updateCoveredStatistic(intp statisticIndex, uint32 weight, bool remove)

        AbstractRefinementSearch* beginSearch(intp numLabelIndices, const intp* labelIndices)

        void applyPrediction(intp statisticIndex, const intp* labelIndices, HeadCandidate* head)


cdef class ExampleWiseStatistics(GradientStatistics):
    pass
