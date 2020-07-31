from boomer.common._arrays cimport intp, float64
from boomer.common.input_data cimport LabelMatrix, AbstractLabelMatrix
from boomer.common.rule_evaluation cimport DefaultPrediction, LabelWisePrediction, DefaultRuleEvaluation, \
    AbstractDefaultRuleEvaluation
from boomer.boosting.label_wise_losses cimport LabelWiseLoss, AbstractLabelWiseLoss

from libcpp cimport bool


cdef extern from "cpp/label_wise_rule_evaluation.h" namespace "boosting":

    cdef cppclass LabelWiseDefaultRuleEvaluationImpl(AbstractDefaultRuleEvaluation):

        # Constructors:

        LabelWiseDefaultRuleEvaluationImpl(AbstractLabelWiseLoss* lossFunction, float64 l2RegularizationWeight) except +

        # Functions:

        DefaultPrediction* calculateDefaultPrediction(AbstractLabelMatrix* labelMatrix) nogil


    cdef cppclass LabelWiseRuleEvaluationImpl:

        # Constructors:

        LabelWiseRuleEvaluationImpl(float64 l2RegularizationWeight) except +

        # Functions:

        void calculateLabelWisePrediction(const intp* labelIndices, const float64* totalSumsOfGradients,
                                          const float64* sumsOfGradients, const float64* totalSumsOfHessians,
                                          const float64* sumsOfHessians, bool uncovered,
                                          LabelWisePrediction* prediction) nogil


cdef class LabelWiseDefaultRuleEvaluation(DefaultRuleEvaluation):

    # Attributes:

    cdef AbstractDefaultRuleEvaluation* default_rule_evaluation

    # Functions:

    cdef DefaultPrediction* calculate_default_prediction(self, LabelMatrix label_matrix)


cdef class LabelWiseRuleEvaluation:

    # Attributes:

    cdef LabelWiseRuleEvaluationImpl* rule_evaluation

    # Functions:

    cdef void calculate_label_wise_prediction(self, const intp[::1] label_indices,
                                              const float64[::1] total_sums_of_gradients,
                                              float64[::1] sums_of_gradients, const float64[::1] total_sums_of_hessians,
                                              float64[::1] sums_of_hessians, bint uncovered,
                                              LabelWisePrediction* prediction)
