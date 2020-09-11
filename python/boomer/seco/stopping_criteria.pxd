from boomer.common._arrays cimport intp, float64
from boomer.common.statistics cimport AbstractStatistics
from boomer.common.stopping_criteria cimport StoppingCriterion


cdef class UncoveredLabelsCriterion(StoppingCriterion):

    # Attributes:

    cdef readonly float64 threshold

    # Functions:

    cdef bint should_continue(self, AbstractStatistics* statistics, intp num_rules)
