from boomer.algorithm._arrays cimport intp, float64
from boomer.algorithm.coverage_losses cimport CoverageLoss


cdef class StoppingCriterion:

    # Functions:

    cpdef bint should_continue(self, list theory)


cdef class SizeStoppingCriterion(StoppingCriterion):

    # Attributes:

    cdef readonly intp max_rules

    # Functions:

    cpdef bint should_continue(self, list theory)


cdef class TimeStoppingCriterion(StoppingCriterion):

    # Attributes:

    cdef readonly intp time_limit

    cdef intp start_time

    # Functions:

    cpdef bint should_continue(self, list theory)


cdef class UncoveredLabelsCriterion(StoppingCriterion):

    # Attributes:

    cdef readonly float64 threshold

    cdef CoverageLoss loss

    # Functions:

    cpdef bint should_continue(self, list theory)
