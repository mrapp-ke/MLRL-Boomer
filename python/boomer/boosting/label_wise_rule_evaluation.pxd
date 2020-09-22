from boomer.common._arrays cimport float64

from libcpp.memory cimport shared_ptr


cdef extern from "cpp/label_wise_rule_evaluation.h" namespace "boosting" nogil:

    cdef cppclass ILabelWiseRuleEvaluation:
        pass


    cdef cppclass RegularizedLabelWiseRuleEvaluationImpl(ILabelWiseRuleEvaluation):

        # Constructors:

        RegularizedLabelWiseRuleEvaluationImpl(float64 l2RegularizationWeight) except +


cdef class LabelWiseRuleEvaluation:

    # Attributes:

    cdef shared_ptr[ILabelWiseRuleEvaluation] rule_evaluation_ptr


cdef class RegularizedLabelWiseRuleEvaluation(LabelWiseRuleEvaluation):
    pass
