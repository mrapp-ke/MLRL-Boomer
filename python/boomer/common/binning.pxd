from boomer.common._arrays cimport uint32


cdef extern from "cpp/binning.h" nogil:

    cdef cppclass IBinning:

        # Constructors:

            IBinning();

    cdef cppclass EqualFrequencyBinningImpl:

        # Constructors:

            EqualFrequencyBinningImpl();


    cdef cppclass EqualWidthBinningImpl:

        # Constructors:

            EqualFrequencyBinningImpl();
