from boomer.common.head_refinement cimport HeadCandidate


cdef class PostProcessor:

    # Functions:

    cdef void post_process(self, HeadCandidate* head)
