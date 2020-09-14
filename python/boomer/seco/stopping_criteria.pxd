from boomer.common._arrays cimport uint32, float64
from boomer.common.statistics cimport AbstractStatistics
from boomer.common.stopping_criteria cimport StoppingCriterion


cdef class UncoveredLabelsCriterion(StoppingCriterion):

    # Attributes:

    cdef readonly float64 threshold

    # Functions:

    cdef bint should_continue(self, AbstractStatistics* statistics, uint32 num_rules)
