from boomer.seco.heuristics cimport AbstractHeuristic

from libcpp cimport bool
from libcpp.memory cimport shared_ptr


cdef extern from "cpp/label_wise_rule_evaluation.h" namespace "seco" nogil:

    cdef cppclass AbstractLabelWiseRuleEvaluation:
        pass

    cdef cppclass HeuristicLabelWiseRuleEvaluationImpl(AbstractLabelWiseRuleEvaluation):

        # Constructors:

        HeuristicLabelWiseRuleEvaluationImpl(AbstractHeuristic* heuristic, bool predictMajority) except +


cdef class LabelWiseRuleEvaluation:

    # Attributes:

    cdef shared_ptr[AbstractLabelWiseRuleEvaluation] rule_evaluation_ptr


cdef class HeuristicLabelWiseRuleEvaluation(LabelWiseRuleEvaluation):
    pass
