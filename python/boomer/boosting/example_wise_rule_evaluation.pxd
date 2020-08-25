from boomer.common._arrays cimport intp, float64
from boomer.common._predictions cimport Prediction, PredictionCandidate, LabelWisePredictionCandidate
from boomer.common.input_data cimport AbstractLabelMatrix
from boomer.common.rule_evaluation cimport DefaultRuleEvaluation, AbstractDefaultRuleEvaluation
from boomer.boosting._blas cimport Blas
from boomer.boosting._lapack cimport Lapack
from boomer.boosting.example_wise_losses cimport AbstractExampleWiseLoss

from libcpp cimport bool
from libcpp.memory cimport shared_ptr


cdef extern from "cpp/example_wise_rule_evaluation.h" namespace "boosting" nogil:

    cdef cppclass ExampleWiseDefaultRuleEvaluationImpl(AbstractDefaultRuleEvaluation):

        # Constructors:

        ExampleWiseDefaultRuleEvaluationImpl(shared_ptr[AbstractExampleWiseLoss] lossFunctionPtr,
                                             float64 l2RegularizationWeight, shared_ptr[Lapack] lapackPtr) except +

        Prediction* calculateDefaultPrediction(AbstractLabelMatrix* labelMatrix) except +


    cdef cppclass ExampleWiseRuleEvaluationImpl:

        # Constructors:

        ExampleWiseRuleEvaluationImpl(float64 l2RegularizationWeight, shared_ptr[Blas] blasPtr,
                                      shared_ptr[Lapack] lapackPtr) except +

        # Functions:

        void calculateLabelWisePrediction(const intp* labelIndices, const float64* totalSumsOfGradients,
                                          float64* sumsOfGradients, const float64* totalSumsOfHessians,
                                          float64* sumsOfHessians, bool uncovered,
                                          LabelWisePredictionCandidate* prediction) except +

        void calculateExampleWisePrediction(const intp* labelIndices, const float64* totalSumsOfGradients,
                                            float64* sumsOfGradients, const float64* totalSumsOfHessians,
                                            float64* sumsOfHessians, float64* tmpGradients, float64* tmpHessians,
                                            int dsysvLwork, float64* dsysvTmpArray1, int* dsysvTmpArray2,
                                            double* dsysvTmpArray3, float64* dspmvTmpArray, bool uncovered,
                                            PredictionCandidate* prediction) except +


cdef class ExampleWiseDefaultRuleEvaluation(DefaultRuleEvaluation):
    pass


cdef class ExampleWiseRuleEvaluation:

    # Attributes:

    cdef shared_ptr[ExampleWiseRuleEvaluationImpl] rule_evaluation_ptr
