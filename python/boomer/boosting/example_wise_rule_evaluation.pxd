from boomer.common._arrays cimport intp, float64
from boomer.common.input_data cimport LabelMatrix, AbstractLabelMatrix
from boomer.common.rule_evaluation cimport DefaultPrediction, Prediction, LabelWisePrediction, DefaultRuleEvaluation, \
    AbstractDefaultRuleEvaluation
from boomer.boosting._blas cimport Blas
from boomer.boosting._lapack cimport Lapack
from boomer.boosting.example_wise_losses cimport AbstractExampleWiseLoss

from libcpp cimport bool


cdef extern from "cpp/example_wise_rule_evaluation.h" namespace "boosting" nogil:

    cdef cppclass ExampleWiseDefaultRuleEvaluationImpl(AbstractDefaultRuleEvaluation):

        # Constructors:

        ExampleWiseDefaultRuleEvaluationImpl(AbstractExampleWiseLoss* lossFunction,
                                             float64 l2RegularizationWeight, Lapack* lapack) except +

        DefaultPrediction* calculateDefaultPrediction(AbstractLabelMatrix* labelMatrix) except +


    cdef cppclass ExampleWiseRuleEvaluationImpl:

        # Constructors:

        ExampleWiseRuleEvaluationImpl(float64 l2RegularizationWeight, Blas* blas, Lapack* lapack) except +

        # Functions:

        void calculateLabelWisePrediction(const intp* labelIndices, const float64* totalSumsOfGradients,
                                          float64* sumsOfGradients, const float64* totalSumsOfHessians,
                                          float64* sumsOfHessians, bool uncovered,
                                          LabelWisePrediction* prediction) except +

        void calculateExampleWisePrediction(const intp* labelIndices, const float64* totalSumsOfGradients,
                                            float64* sumsOfGradients, const float64* totalSumsOfHessians,
                                            float64* sumsOfHessians, bool uncovered,
                                            Prediction* prediction) except +


cdef class ExampleWiseDefaultRuleEvaluation(DefaultRuleEvaluation):

    # Functions:

    cdef DefaultPrediction* calculate_default_prediction(self, LabelMatrix label_matrix) nogil


cdef class ExampleWiseRuleEvaluation:

    # Attributes:

    cdef ExampleWiseRuleEvaluationImpl* rule_evaluation
