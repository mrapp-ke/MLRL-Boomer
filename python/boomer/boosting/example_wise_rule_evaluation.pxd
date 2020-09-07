from boomer.common._arrays cimport intp, float64
from boomer.common._predictions cimport PredictionCandidate, LabelWisePredictionCandidate
from boomer.boosting._blas cimport Blas
from boomer.boosting._lapack cimport Lapack

from libcpp cimport bool
from libcpp.memory cimport shared_ptr


cdef extern from "cpp/example_wise_rule_evaluation.h" namespace "boosting" nogil:

    cdef cppclass AbstractExampleWiseRuleEvaluation:

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


    cdef cppclass RegularizedExampleWiseRuleEvaluationImpl(AbstractExampleWiseRuleEvaluation):

        # Constructors:

        RegularizedExampleWiseRuleEvaluationImpl(float64 l2RegularizationWeight, shared_ptr[Blas] blasPtr,
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


cdef class ExampleWiseRuleEvaluation:

    # Attributes:

    cdef shared_ptr[AbstractExampleWiseRuleEvaluation] rule_evaluation_ptr


cdef class RegularizedExampleWiseRuleEvaluation(ExampleWiseRuleEvaluation):
    pass
