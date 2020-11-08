from boomer.common._types cimport uint32


cdef extern from "cpp/random.h" nogil:

    cdef cppclass RNG:

        # Constructors:

        RNG(uint32 randomState) except +

        # Functions:

        uint32 random(uint32 min, uint32 max)
