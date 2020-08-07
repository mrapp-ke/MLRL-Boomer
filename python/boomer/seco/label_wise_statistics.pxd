from boomer.common._arrays cimport intp, uint8, uint32, float64
from boomer.common.input_data cimport LabelMatrix, AbstractLabelMatrix
from boomer.common.statistics cimport AbstractRefinementSearch, AbstractDecomposableRefinementSearch
from boomer.common.head_refinement cimport HeadCandidate
from boomer.common.rule_evaluation cimport DefaultPrediction, Prediction, LabelWisePrediction
from boomer.seco.statistics cimport CoverageStatistics, AbstractCoverageStatistics
from boomer.seco.label_wise_rule_evaluation cimport LabelWiseRuleEvaluation, LabelWiseRuleEvaluationImpl

from libcpp cimport bool


cdef extern from "cpp/label_wise_statistics.h" namespace "seco" nogil:

    cdef cppclass LabelWiseRefinementSearchImpl(AbstractDecomposableRefinementSearch):

        # Constructors:

        LabelWiseRefinementSearchImpl(LabelWiseRuleEvaluationImpl* ruleEvaluation, intp numLabels,
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

        LabelWiseStatisticsImpl(LabelWiseRuleEvaluationImpl* ruleEvaluation) except +

        # Attributes:

        float64 sumUncoveredLabels_;

        # Functions:

        void applyDefaultPrediction(AbstractLabelMatrix* labelMatrix, DefaultPrediction* defaultPrediction)

        void resetSampledStatistics()

        void addSampledStatistic(intp statisticIndex, uint32 weight)

        void resetCoveredStatistics()

        void updateCoveredStatistic(intp statisticIndex, uint32 weight, bool remove)

        AbstractRefinementSearch* beginSearch(intp numPredictions, const intp* labelIndices)

        void applyPrediction(intp statisticIndex, const intp* labelIndices, HeadCandidate* head)


cdef class LabelWiseStatistics(CoverageStatistics):

    # Attributes:

    cdef LabelWiseRuleEvaluation rule_evaluation

    cdef LabelMatrix label_matrix

    cdef float64[:, ::1] uncovered_labels

    cdef uint8[::1] minority_labels

    cdef float64[:, ::1] confusion_matrices_total

    cdef float64[:, ::1] confusion_matrices_subset

    # Functions:

    cdef void apply_default_prediction(self, LabelMatrix label_matrix, DefaultPrediction* default_prediction)

    cdef void reset_sampled_statistics(self)

    cdef void add_sampled_statistic(self, intp statistic_index, uint32 weight)

    cdef void reset_covered_statistics(self)

    cdef void update_covered_statistic(self, intp statistic_index, uint32 weight, bint remove)

    cdef AbstractRefinementSearch* begin_search(self, intp[::1] label_indices)

    cdef void apply_prediction(self, intp statistic_index, intp[::1] label_indices, HeadCandidate* head)
