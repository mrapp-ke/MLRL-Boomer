from boomer.common._arrays cimport float64
from boomer.common.head_refinement cimport HeadCandidate


cdef class Shrinkage:

    # Functions:

    cdef void apply_shrinkage(self, HeadCandidate* head)


cdef class ConstantShrinkage(Shrinkage):

    # Attributes:

    cdef float64 shrinkage

    # Functions:

    cdef void apply_shrinkage(self, HeadCandidate* head)
