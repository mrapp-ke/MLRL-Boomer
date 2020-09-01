from boomer.common._arrays cimport intp, float64
from boomer.common._predictions cimport LabelWisePredictionCandidate

from libcpp cimport bool
from libcpp.memory cimport shared_ptr


cdef extern from "cpp/label_wise_rule_evaluation.h" namespace "boosting" nogil:

    cdef cppclass LabelWiseRuleEvaluationImpl:

        # Constructors:

        LabelWiseRuleEvaluationImpl(float64 l2RegularizationWeight) except +

        # Functions:

        void calculateLabelWisePrediction(const intp* labelIndices, const float64* totalSumsOfGradients,
                                          float64* sumsOfGradients, const float64* totalSumsOfHessians,
                                          float64* sumsOfHessians, bool uncovered,
                                          LabelWisePredictionCandidate* prediction) except +


cdef class LabelWiseRuleEvaluation:

    # Attributes:

    cdef shared_ptr[LabelWiseRuleEvaluationImpl] rule_evaluation_ptr
