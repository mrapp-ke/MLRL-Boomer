from boomer.common._arrays cimport intp, float64
from boomer.common._predictions cimport LabelWisePredictionCandidate

from libcpp cimport bool
from libcpp.memory cimport shared_ptr


cdef extern from "cpp/label_wise_rule_evaluation.h" namespace "boosting" nogil:

    cdef cppclass AbstractLabelWiseRuleEvaluation:

        # Functions:

        void calculateLabelWisePrediction(const intp* labelIndices, const float64* totalSumsOfGradients,
                                          float64* sumsOfGradients, const float64* totalSumsOfHessians,
                                          float64* sumsOfHessians, bool uncovered,
                                          LabelWisePredictionCandidate* prediction) except +


    cdef cppclass RegularizedLabelWiseRuleEvaluationImpl(AbstractLabelWiseRuleEvaluation):

        # Constructors:

        RegularizedLabelWiseRuleEvaluationImpl(float64 l2RegularizationWeight) except +

        # Functions:

        void calculateLabelWisePrediction(const intp* labelIndices, const float64* totalSumsOfGradients,
                                          float64* sumsOfGradients, const float64* totalSumsOfHessians,
                                          float64* sumsOfHessians, bool uncovered,
                                          LabelWisePredictionCandidate* prediction) except +


cdef class LabelWiseRuleEvaluation:

    # Attributes:

    cdef shared_ptr[AbstractLabelWiseRuleEvaluation] rule_evaluation_ptr


cdef class RegularizedLabelWiseRuleEvaluation(LabelWiseRuleEvaluation):
    pass
