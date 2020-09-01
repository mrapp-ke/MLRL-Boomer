"""
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides Cython wrappers for C++ classes that allow to calculate the predictions of rules, as well as corresponding
quality scores.
"""
from boomer.seco.heuristics cimport Heuristic

from libcpp.memory cimport make_shared


cdef class LabelWiseRuleEvaluation:
    """
    A wrapper for the abstract C++ class `AbstractLabelWiseRuleEvaluation`.
    """
    pass


cdef class HeuristicLabelWiseRuleEvaluation(LabelWiseRuleEvaluation):
    """
    A wrapper for the C++ class `HeuristicLabelWiseRuleEvaluationImpl`.
    """

    def __cinit__(self, Heuristic heuristic):
        """
        :param heuristic: The heuristic that should be used
        """
        self.rule_evaluation_ptr = <shared_ptr[AbstractLabelWiseRuleEvaluation]>make_shared[HeuristicLabelWiseRuleEvaluationImpl](
            heuristic.heuristic_ptr)
