from boomer.common._arrays cimport intp, float64
from boomer.common.statistics cimport AbstractStatistics
from boomer.common.stopping_criteria cimport StoppingCriterion

from libcpp.memory cimport shared_ptr


cdef class UncoveredLabelsCriterion(StoppingCriterion):

    # Attributes:

    cdef readonly float64 threshold

    cdef shared_ptr[AbstractStatistics] statistics_ptr

    # Functions:

    cdef bint should_continue(self, intp num_rules)
