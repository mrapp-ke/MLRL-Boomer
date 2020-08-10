from boomer.common._arrays cimport uint32, intp, float64
from boomer.common.input_data cimport AbstractLabelMatrix
from boomer.common.statistics cimport AbstractStatistics, AbstractRefinementSearch, \
    AbstractDecomposableRefinementSearch
from boomer.common.head_refinement cimport HeadCandidate
from boomer.common.rule_evaluation cimport DefaultPrediction, Prediction, LabelWisePrediction
from boomer.boosting.statistics cimport GradientStatistics, AbstractGradientStatistics
from boomer.boosting.label_wise_losses cimport AbstractLabelWiseLoss
from boomer.boosting.label_wise_rule_evaluation cimport LabelWiseRuleEvaluationImpl

from libcpp cimport bool
from libcpp.memory cimport shared_ptr


cdef extern from "cpp/label_wise_statistics.h" namespace "boosting" nogil:

    cdef cppclass LabelWiseRefinementSearchImpl(AbstractDecomposableRefinementSearch):

        # Constructors:

        LabelWiseRefinementSearchImpl(shared_ptr[LabelWiseRuleEvaluationImpl] ruleEvaluationPtr, intp numPredictions,
                                      const intp* labelIndices, intp numLabels, const float64* gradients,
                                      const float64* totalSumsOfGradients, const float64* hessians,
                                      const float64* totalSumsOfHessians) except +

        # Functions:

        void updateSearch(intp statisticIndex, uint32 weight)

        void resetSearch()

        LabelWisePrediction* calculateLabelWisePrediction(bool uncovered, bool accumulated) except +

        Prediction* calculateExampleWisePrediction(bool uncovered, bool accumulated) except +


    cdef cppclass LabelWiseStatisticsImpl(AbstractGradientStatistics):

        # Constructors:

        LabelWiseStatisticsImpl(shared_ptr[AbstractLabelWiseLoss] lossFunctionPtr,
                                shared_ptr[LabelWiseRuleEvaluationImpl] ruleEvaluationPtr) except +

        # Functions:

        void applyDefaultPrediction(AbstractLabelMatrix* labelMatrix, DefaultPrediction* defaultPrediction)

        void resetSampledStatistics()

        void addSampledStatistic(intp statisticIndex, uint32 weight)

        void resetCoveredStatistics()

        void updateCoveredStatistic(intp statisticIndex, uint32 weight, bool remove)

        AbstractRefinementSearch* beginSearch(intp numLabelIndices, const intp* labelIndices)

        void applyPrediction(intp statisticIndex, const intp* labelIndices, HeadCandidate* head)


cdef class LabelWiseStatistics(GradientStatistics):
    pass
