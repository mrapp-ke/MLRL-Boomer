from boomer.common._arrays cimport intp, float64
from boomer.common.stopping_criteria cimport StoppingCriterion
from boomer.seco.coverage_losses cimport CoverageLoss


cdef class UncoveredLabelsCriterion(StoppingCriterion):

    # Attributes:

    cdef readonly float64 threshold

    cdef CoverageLoss loss

    # Functions:

    cdef bint should_continue(self, intp num_rules)
