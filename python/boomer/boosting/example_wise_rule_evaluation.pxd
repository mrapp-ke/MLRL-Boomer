from boomer.common._arrays cimport intp, float64
from boomer.common.input_data cimport AbstractLabelMatrix
from boomer.common.rule_evaluation cimport DefaultPrediction, Prediction, LabelWisePrediction, DefaultRuleEvaluation, \
    AbstractDefaultRuleEvaluation
from boomer.boosting._blas cimport Blas
from boomer.boosting._lapack cimport Lapack
from boomer.boosting.example_wise_losses cimport AbstractExampleWiseLoss

from libcpp cimport bool
from libcpp.memory cimport shared_ptr


cdef extern from "cpp/example_wise_rule_evaluation.h" namespace "boosting" nogil:

    cdef cppclass ExampleWiseDefaultRuleEvaluationImpl(AbstractDefaultRuleEvaluation):

        # Constructors:

        ExampleWiseDefaultRuleEvaluationImpl(shared_ptr[AbstractExampleWiseLoss] lossFunctionPtr,
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
    pass


cdef class ExampleWiseRuleEvaluation:

    # Attributes:

    cdef shared_ptr[ExampleWiseRuleEvaluationImpl] rule_evaluation_ptr
