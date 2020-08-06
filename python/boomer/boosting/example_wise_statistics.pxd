from boomer.common._arrays cimport uint32, intp, float64
from boomer.common.input_data cimport LabelMatrix
from boomer.common.statistics cimport RefinementSearch, AbstractRefinementSearch
from boomer.common.head_refinement cimport HeadCandidate
from boomer.common.rule_evaluation cimport DefaultPrediction, Prediction, LabelWisePrediction
from boomer.boosting.statistics cimport GradientStatistics
from boomer.boosting.example_wise_losses cimport ExampleWiseLoss
from boomer.boosting.example_wise_rule_evaluation cimport ExampleWiseRuleEvaluation, ExampleWiseRuleEvaluationImpl

from libcpp cimport bool


cdef extern from "cpp/example_wise_statistics.h" namespace "boosting":

    cdef cppclass ExampleWiseRefinementSearchImpl(AbstractRefinementSearch):

        # Constructors:

        ExampleWiseRefinementSearchImpl(ExampleWiseRuleEvaluationImpl* ruleEvaluation, intp numPredictions,
                                        const intp* labelIndices, intp numLabels, const float64* gradients,
                                        const float64* totalSumsOfGradients, const float64* hessians,
                                        const float64* totalSumsOfHessians) except +

        # Functions:

        void updateSearch(intp statisticIndex, uint32 weight) nogil

        void resetSearch() nogil

        LabelWisePrediction* calculateLabelWisePrediction(bool uncovered, bool accumulated) nogil except +

        Prediction* calculateExampleWisePrediction(bool uncovered, bool accumulated) nogil except +


cdef class ExampleWiseRefinementSearch(RefinementSearch):

    # Attributes:

    cdef AbstractRefinementSearch* refinement_search

    # Functions:

    cdef void update_search(self, intp statistic_index, uint32 weight)

    cdef void reset_search(self)

    cdef LabelWisePrediction* calculate_label_wise_prediction(self, bint uncovered, bint accumulated)

    cdef Prediction* calculate_example_wise_prediction(self, bint uncovered, bint accumulated)


cdef class ExampleWiseStatistics(GradientStatistics):

    # Attributes:

    cdef ExampleWiseLoss loss_function

    cdef ExampleWiseRuleEvaluation rule_evaluation

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

    cdef RefinementSearch begin_search(self, intp[::1] label_indices)

    cdef void apply_prediction(self, intp statistic_index, intp[::1] label_indices, HeadCandidate* head)
