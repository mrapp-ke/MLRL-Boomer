from boomer.common._arrays cimport uint32, intp, float64
from boomer.common.input_data cimport LabelMatrix, AbstractLabelMatrix
from boomer.common.statistics cimport AbstractRefinementSearch, AbstractDecomposableRefinementSearch
from boomer.common.head_refinement cimport HeadCandidate
from boomer.common.rule_evaluation cimport DefaultPrediction, Prediction, LabelWisePrediction
from boomer.boosting.statistics cimport GradientStatistics, AbstractGradientStatistics
from boomer.boosting.label_wise_losses cimport LabelWiseLoss, AbstractLabelWiseLoss
from boomer.boosting.label_wise_rule_evaluation cimport LabelWiseRuleEvaluation, LabelWiseRuleEvaluationImpl

from libcpp cimport bool
from libcpp.memory cimport shared_ptr


cdef extern from "cpp/label_wise_statistics.h" namespace "boosting" nogil:

    cdef cppclass LabelWiseRefinementSearchImpl(AbstractDecomposableRefinementSearch):

        # Constructors:

        LabelWiseRefinementSearchImpl(LabelWiseRuleEvaluationImpl* ruleEvaluation, intp numPredictions,
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

    # Attributes:

    cdef LabelWiseLoss loss_function

    cdef LabelWiseRuleEvaluation rule_evaluation

    cdef LabelMatrix label_matrix

    cdef float64[:, ::1] current_scores

    cdef float64[:, ::1] gradients

    cdef float64[::1] total_sums_of_gradients

    cdef float64[:, ::1] hessians

    cdef float64[::1] total_sums_of_hessians

    # Functions:

    cdef void apply_default_prediction(self, LabelMatrix label_matrix, DefaultPrediction* default_prediction)

    cdef void reset_sampled_statistics(self)

    cdef void add_sampled_statistic(self, intp statistic_index, uint32 weight)

    cdef void reset_covered_statistics(self)

    cdef void update_covered_statistic(self, intp statistic_index, uint32 weight, bint remove)

    cdef AbstractRefinementSearch* begin_search(self, intp[::1] label_indices)

    cdef void apply_prediction(self, intp statistic_index, intp[::1] label_indices, HeadCandidate* head)
