from mlrl.seco.cython.heuristics cimport IHeuristic

from libcpp.memory cimport unique_ptr


cdef extern from "seco/rule_evaluation/rule_evaluation_label_wise.hpp" namespace "seco" nogil:

    cdef cppclass ILabelWiseRuleEvaluationFactory:
        pass


cdef extern from "seco/rule_evaluation/rule_evaluation_label_wise_majority.hpp" namespace "seco" nogil:

    cdef cppclass LabelWiseMajorityRuleEvaluationFactoryImpl"seco::LabelWiseMajorityRuleEvaluationFactory"(
            ILabelWiseRuleEvaluationFactory):
        pass


cdef extern from "seco/rule_evaluation/rule_evaluation_label_wise_single.hpp" namespace "seco" nogil:

    cdef cppclass LabelWiseSingleLabelRuleEvaluationFactoryImpl"seco::LabelWiseSingleLabelRuleEvaluationFactory"(
            ILabelWiseRuleEvaluationFactory):

        # Constructors:

        LabelWiseSingleLabelRuleEvaluationFactoryImpl(unique_ptr[IHeuristic] heuristicPtr) except +


cdef class LabelWiseRuleEvaluationFactory:

    # Attributes:

    cdef unique_ptr[ILabelWiseRuleEvaluationFactory] rule_evaluation_factory_ptr


cdef class LabelWiseMajorityRuleEvaluationFactory(LabelWiseRuleEvaluationFactory):
    pass


cdef class LabelWiseSingleLabelRuleEvaluationFactory(LabelWiseRuleEvaluationFactory):
    pass
