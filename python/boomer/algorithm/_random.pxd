from boomer.algorithm._arrays cimport uint32


cdef class RNG:

    # Attributes:

    cdef readonly uint32 random_state

    # Functions:

    cpdef uint32 random(self, uint32 min, uint32 max)
