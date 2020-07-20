from boomer.common._arrays cimport float64
from boomer.common.head_refinement cimport HeadCandidate
from boomer.common.post_processing cimport PostProcessor


cdef class Shrinkage(PostProcessor):

    # Functions:

    cdef void post_process(self, HeadCandidate* head)


cdef class ConstantShrinkage(Shrinkage):

    # Attributes:

    cdef float64 shrinkage

    # Functions:

    cdef void post_process(self, HeadCandidate* head)
