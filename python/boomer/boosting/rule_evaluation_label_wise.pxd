from boomer.common._arrays cimport float64

from libcpp.memory cimport shared_ptr


cdef extern from "cpp/rule_evaluation_label_wise.h" namespace "boosting" nogil:

    cdef cppclass ILabelWiseRuleEvaluationFactory:
        pass


    cdef cppclass RegularizedLabelWiseRuleEvaluationFactoryImpl(ILabelWiseRuleEvaluationFactory):

        # Constructors:

        RegularizedLabelWiseRuleEvaluationFactoryImpl(float64 l2RegularizationWeight) except +


cdef class LabelWiseRuleEvaluationFactory:

    # Attributes:

    cdef shared_ptr[ILabelWiseRuleEvaluationFactory] rule_evaluation_factory_ptr


cdef class RegularizedLabelWiseRuleEvaluationFactory(LabelWiseRuleEvaluationFactory):
    pass
