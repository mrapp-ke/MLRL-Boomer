from boomer.common._arrays cimport intp
from boomer.common.statistics cimport AbstractStatistics


cdef class StoppingCriterion:

    # Functions:

    cdef bint should_continue(self, AbstractStatistics* statistics, intp num_rules)


cdef class SizeStoppingCriterion(StoppingCriterion):

    # Attributes:

    cdef readonly intp max_rules

    # Functions:

    cdef bint should_continue(self, AbstractStatistics* statistics, intp num_rules)


cdef class TimeStoppingCriterion(StoppingCriterion):

    # Attributes:

    cdef readonly intp time_limit

    cdef intp start_time

    # Functions:

    cdef bint should_continue(self, AbstractStatistics* statistics, intp num_rules)
