from boomer.common._types cimport uint32
from boomer.common.statistics cimport IStatistics


cdef class StoppingCriterion:

    # Functions:

    cdef bint should_continue(self, IStatistics* statistics, uint32 num_rules)


cdef class SizeStoppingCriterion(StoppingCriterion):

    # Attributes:

    cdef readonly uint32 max_rules

    # Functions:

    cdef bint should_continue(self, IStatistics* statistics, uint32 num_rules)


cdef class TimeStoppingCriterion(StoppingCriterion):

    # Attributes:

    cdef readonly uint32 time_limit

    cdef long start_time

    # Functions:

    cdef bint should_continue(self, IStatistics* statistics, uint32 num_rules)
