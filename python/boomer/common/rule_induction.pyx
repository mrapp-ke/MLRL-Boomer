"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)
"""
from libcpp.memory cimport make_shared


cdef class RuleInduction:
    """
    A wrapper for the pure virtual C++ class `IRuleInduction`.
    """
    pass


cdef class TopDownRuleInduction(RuleInduction):
    """
    A wrapper for the C++ class `TopDownRuleInduction`.
    """

    def __cinit__(self, uint32 num_threads):
        """
        :param num_threads: The number of CPU threads to be used to search for potential refinements of a rule in
                            parallel. Must be at least 1
        """
        self.rule_induction_ptr = <shared_ptr[IRuleInduction]>make_shared[TopDownRuleInductionImpl](num_threads)
