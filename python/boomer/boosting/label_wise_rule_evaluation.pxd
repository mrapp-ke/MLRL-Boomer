from boomer.common._arrays cimport intp, float64
from boomer.common.input_data cimport LabelMatrix, AbstractLabelMatrix
from boomer.common.rule_evaluation cimport DefaultPrediction, LabelWisePrediction, DefaultRuleEvaluation, \
    AbstractDefaultRuleEvaluation
from boomer.boosting.label_wise_losses cimport AbstractLabelWiseLoss

from libcpp cimport bool


cdef extern from "cpp/label_wise_rule_evaluation.h" namespace "boosting" nogil:

    cdef cppclass LabelWiseDefaultRuleEvaluationImpl(AbstractDefaultRuleEvaluation):

        # Constructors:

        LabelWiseDefaultRuleEvaluationImpl(AbstractLabelWiseLoss* lossFunction, float64 l2RegularizationWeight) except +

        # Functions:

        DefaultPrediction* calculateDefaultPrediction(AbstractLabelMatrix* labelMatrix) except +


    cdef cppclass LabelWiseRuleEvaluationImpl:

        # Constructors:

        LabelWiseRuleEvaluationImpl(float64 l2RegularizationWeight) except +

        # Functions:

        void calculateLabelWisePrediction(const intp* labelIndices, const float64* totalSumsOfGradients,
                                          float64* sumsOfGradients, const float64* totalSumsOfHessians,
                                          float64* sumsOfHessians, bool uncovered,
                                          LabelWisePrediction* prediction) except +


cdef class LabelWiseDefaultRuleEvaluation(DefaultRuleEvaluation):

    # Functions:

    cdef DefaultPrediction* calculate_default_prediction(self, LabelMatrix label_matrix) nogil


cdef class LabelWiseRuleEvaluation:

    # Attributes:

    cdef LabelWiseRuleEvaluationImpl* rule_evaluation
