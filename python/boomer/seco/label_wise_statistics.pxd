from boomer.common._arrays cimport intp, uint8, uint32, float64
from boomer.common.input_data cimport LabelMatrix, AbstractLabelMatrix
from boomer.common.statistics cimport AbstractStatistics, AbstractRefinementSearch, AbstractDecomposableRefinementSearch
from boomer.common.head_refinement cimport HeadCandidate
from boomer.common.rule_evaluation cimport DefaultPrediction, Prediction, LabelWisePrediction
from boomer.seco.statistics cimport CoverageStatistics, AbstractCoverageStatistics
from boomer.seco.label_wise_rule_evaluation cimport LabelWiseRuleEvaluationImpl

from libcpp cimport bool
from libcpp.memory cimport shared_ptr


cdef extern from "cpp/label_wise_statistics.h" namespace "seco" nogil:

    cdef cppclass LabelWiseRefinementSearchImpl(AbstractDecomposableRefinementSearch):

        # Constructors:

        LabelWiseRefinementSearchImpl(shared_ptr[LabelWiseRuleEvaluationImpl] ruleEvaluationPtr, intp numLabels,
                                      const intp* labelIndices, AbstractLabelMatrix* labelMatrix,
                                      const float64* uncoveredLabels, const uint8* minorityLabels,
                                      const float64* confusionMatricesTotal,
                                      const float64* confusionMatricesSubset) except +

        # Functions:

        void updateSearch(intp statisticIndex, uint32 weight)

        void resetSearch()

        LabelWisePrediction* calculateLabelWisePrediction(bool uncovered, bool accumulated) except +

        Prediction* calculateExampleWisePrediction(bool uncovered, bool accumulated) except +


    cdef cppclass LabelWiseStatisticsImpl(AbstractCoverageStatistics):

        # Constructors:

        LabelWiseStatisticsImpl(shared_ptr[LabelWiseRuleEvaluationImpl] ruleEvaluationPtr) except +

        # Attributes:

        float64 sumUncoveredLabels_;

        # Functions:

        void applyDefaultPrediction(AbstractLabelMatrix* labelMatrix, DefaultPrediction* defaultPrediction)

        void resetSampledStatistics()

        void addSampledStatistic(intp statisticIndex, uint32 weight)

        void resetCoveredStatistics()

        void updateCoveredStatistic(intp statisticIndex, uint32 weight, bool remove)

        AbstractRefinementSearch* beginSearch(intp numLabelIndices, const intp* labelIndices)

        void applyPrediction(intp statisticIndex, const intp* labelIndices, HeadCandidate* head)


cdef class LabelWiseStatistics(CoverageStatistics):
    pass
