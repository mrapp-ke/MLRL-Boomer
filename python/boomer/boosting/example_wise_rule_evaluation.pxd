from boomer.common._arrays cimport intp, float64
from boomer.common.input_data cimport LabelMatrix
from boomer.common.rule_evaluation cimport DefaultPrediction, Prediction, LabelWisePrediction, DefaultRuleEvaluation
from boomer.boosting.example_wise_losses cimport ExampleWiseLoss

from libcpp cimport bool


cdef extern from "cpp/example_wise_rule_evaluation.h" namespace "boosting":

    cdef cppclass ExampleWiseRuleEvaluationImpl:

        # Constructors:

        ExampleWiseRuleEvaluationImpl(float64 l2RegularizationWeight)

        # Functions:

        void calculateLabelWisePrediction(const intp* labelIndices, const float64* totalSumsOfGradients,
                                          float64* sumsOfGradients, const float64* totalSumsOfHessians,
                                          float64* sumsOfHessians, bool uncovered,
                                          LabelWisePrediction* prediction) nogil

        void calculateExampleWisePrediction(const intp* labelIndices, const float64* totalSumsOfGradients,
                                            float64* sumsOfGradients, const float64* totalSumsOfHessians,
                                            float64* sumsOfHessians, bool uncovered, Prediction* prediction) nogil


cdef class ExampleWiseDefaultRuleEvaluation(DefaultRuleEvaluation):

    # Attributes:

    cdef ExampleWiseLoss loss_function

    cdef float64 l2_regularization_weight

    # Functions:

    cdef DefaultPrediction* calculate_default_prediction(self, LabelMatrix label_matrix)


cdef class ExampleWiseRuleEvaluation:

    # Attributes:

    cdef ExampleWiseRuleEvaluationImpl* rule_evaluation

    # Functions:

    cdef void calculate_label_wise_prediction(self, const intp[::1] label_indices,
                                              const float64[::1] total_sums_of_gradients,
                                              float64[::1] sums_of_gradients, const float64[::1] total_sums_of_hessians,
                                              float64[::1] sums_of_hessians, bint uncovered,
                                              LabelWisePrediction* prediction)

    cdef void calculate_example_wise_prediction(self, const intp[::1] label_indices,
                                                const float64[::1] total_sums_of_gradients,
                                                float64[::1] sums_of_gradients,
                                                const float64[::1] total_sums_of_hessians,
                                                float64[::1] sums_of_hessians, bint uncovered,
                                                Prediction* prediction)
