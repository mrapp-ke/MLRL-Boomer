"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)
"""
from libcpp.memory cimport shared_ptr, make_shared


cdef class RuleInduction:
    """
    A wrapper for the pure virtual C++ class `IRuleInduction`.
    """
    pass


cdef class TopDownRuleInduction(RuleInduction):
    """
    A wrapper for the C++ class `TopDownRuleInduction`.
    """

    def __cinit__(self):
        self.rule_induction_ptr = <shared_ptr[IRuleInduction]>make_shared[TopDownRuleInductionImpl]()
