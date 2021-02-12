"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)
"""
from libcpp.memory cimport make_shared


cdef class StoppingCriterion:
    """
    A wrapper for the pure virtual C++ class `IStoppingCriterion`.
    """
    pass


cdef class SizeStoppingCriterion(StoppingCriterion):
    """
    A wrapper for the C++ class `SizeStoppingCriterion`.
    """

    def __cinit__(self, uint32 max_rules):
        """
        :param max_rules: The maximum number of rules
        """
        self.stopping_criterion_ptr = <shared_ptr[IStoppingCriterion]>make_shared[SizeStoppingCriterionImpl](max_rules)


cdef class TimeStoppingCriterion(StoppingCriterion):
    """
    A wrapper for the C++ class `TimeStoppingCriterion`.
    """

    def __cinit__(self, uint32 time_limit):
        """
        :param time_limit: The time limit in seconds
        """
        self.stopping_criterion_ptr = <shared_ptr[IStoppingCriterion]>make_shared[TimeStoppingCriterionImpl](time_limit)
