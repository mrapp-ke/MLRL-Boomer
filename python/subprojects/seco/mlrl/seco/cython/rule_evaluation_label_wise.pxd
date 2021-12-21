from mlrl.seco.cython.heuristics cimport IHeuristicFactory
from mlrl.seco.cython.lift_functions cimport ILiftFunctionFactory

from libcpp.memory cimport unique_ptr


cdef extern from "seco/rule_evaluation/rule_evaluation_label_wise.hpp" namespace "seco" nogil:

    cdef cppclass ILabelWiseRuleEvaluationFactory:
        pass


cdef extern from "seco/rule_evaluation/rule_evaluation_label_wise_majority.hpp" namespace "seco" nogil:

    cdef cppclass LabelWiseMajorityRuleEvaluationFactoryImpl"seco::LabelWiseMajorityRuleEvaluationFactory"(
            ILabelWiseRuleEvaluationFactory):
        pass


cdef extern from "seco/rule_evaluation/rule_evaluation_label_wise_partial.hpp" namespace "seco" nogil:

    cdef cppclass LabelWisePartialRuleEvaluationFactoryImpl"seco::LabelWisePartialRuleEvaluationFactory"(
            ILabelWiseRuleEvaluationFactory):

        # Constructors:

        LabelWisePartialRuleEvaluationFactoryImpl(unique_ptr[IHeuristicFactory] heuristicFactoryPtr,
                                                  unique_ptr[ILiftFunctionFactory] liftFunctionFactoryPtr) except +


cdef extern from "seco/rule_evaluation/rule_evaluation_label_wise_single.hpp" namespace "seco" nogil:

    cdef cppclass LabelWiseSingleLabelRuleEvaluationFactoryImpl"seco::LabelWiseSingleLabelRuleEvaluationFactory"(
            ILabelWiseRuleEvaluationFactory):

        # Constructors:

        LabelWiseSingleLabelRuleEvaluationFactoryImpl(unique_ptr[IHeuristicFactory] heuristicFactoryPtr) except +


cdef class LabelWiseRuleEvaluationFactory:

    # Attributes:

    cdef unique_ptr[ILabelWiseRuleEvaluationFactory] rule_evaluation_factory_ptr


cdef class LabelWiseMajorityRuleEvaluationFactory(LabelWiseRuleEvaluationFactory):
    pass


cdef class LabelWisePartialRuleEvaluationFactory(LabelWiseRuleEvaluationFactory):
    pass


cdef class LabelWiseSingleLabelRuleEvaluationFactory(LabelWiseRuleEvaluationFactory):
    pass
