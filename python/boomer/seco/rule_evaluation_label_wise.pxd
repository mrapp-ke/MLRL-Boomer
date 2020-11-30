from boomer.seco.heuristics cimport IHeuristic

from libcpp cimport bool
from libcpp.memory cimport shared_ptr


cdef extern from "cpp/rule_evaluation/rule_evaluation_label_wise.h" namespace "seco" nogil:

    cdef cppclass ILabelWiseRuleEvaluationFactory:
        pass


cdef extern from "cpp/rule_evaluation/rule_evaluation_label_wise_heuristic.h" namespace "seco" nogil:

    cdef cppclass HeuristicLabelWiseRuleEvaluationFactoryImpl"seco::HeuristicLabelWiseRuleEvaluationFactory"(
            ILabelWiseRuleEvaluationFactory):

        # Constructors:

        HeuristicLabelWiseRuleEvaluationFactoryImpl(IHeuristic* heuristic, bool predictMajority) except +


cdef class LabelWiseRuleEvaluationFactory:

    # Attributes:

    cdef shared_ptr[ILabelWiseRuleEvaluationFactory] rule_evaluation_factory_ptr


cdef class HeuristicLabelWiseRuleEvaluationFactory(LabelWiseRuleEvaluationFactory):
    pass
