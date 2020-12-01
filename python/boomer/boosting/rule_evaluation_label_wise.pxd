from boomer.common._types cimport uint32, float64

from libcpp.memory cimport shared_ptr


cdef extern from "cpp/rule_evaluation/rule_evaluation_label_wise.h" namespace "boosting" nogil:

    cdef cppclass ILabelWiseRuleEvaluationFactory:
        pass


cdef extern from "cpp/rule_evaluation/rule_evaluation_label_wise_regularized.h" namespace "boosting" nogil:

    cdef cppclass RegularizedLabelWiseRuleEvaluationFactoryImpl"boosting::RegularizedLabelWiseRuleEvaluationFactory"(
            ILabelWiseRuleEvaluationFactory):

        # Constructors:

        RegularizedLabelWiseRuleEvaluationFactoryImpl(float64 l2RegularizationWeight) except +


cdef extern from "cpp/rule_evaluation/rule_evaluation_label_wise_binning.h" namespace "boosting" nogil:

    cdef cppclass BinningLabelWiseRuleEvaluationFactoryImpl"boosting::BinningLabelWiseRuleEvaluationFactory"(
            ILabelWiseRuleEvaluationFactory):

        # Constructors:

        BinningLabelWiseRuleEvaluationFactoryImpl(float64 l2RegularizationWeight, uint32 numPositiveBins,
                                                  uint32 numNegativeBins) except +


cdef class LabelWiseRuleEvaluationFactory:

    # Attributes:

    cdef shared_ptr[ILabelWiseRuleEvaluationFactory] rule_evaluation_factory_ptr


cdef class RegularizedLabelWiseRuleEvaluationFactory(LabelWiseRuleEvaluationFactory):
    pass


cdef class BinningLabelWiseRuleEvaluationFactory(LabelWiseRuleEvaluationFactory):
    pass
