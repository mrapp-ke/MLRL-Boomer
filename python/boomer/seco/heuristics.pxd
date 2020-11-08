from boomer.common._types cimport float64

from libcpp.memory cimport shared_ptr


cdef extern from "cpp/heuristics.h" namespace "seco" nogil:

    cdef cppclass IHeuristic:
        pass


    cdef cppclass PrecisionImpl(IHeuristic):
        pass


    cdef cppclass RecallImpl(IHeuristic):
        pass


    cdef cppclass WRAImpl(IHeuristic):
        pass


    cdef cppclass HammingLossImpl(IHeuristic):
        pass


    cdef cppclass FMeasureImpl(IHeuristic):

        # Constructors:

        FMeasureImpl(float64 beta) except +


    cdef cppclass MEstimateImpl(IHeuristic):

        # Constructors:

        MEstimateImpl(float64 m) except +


cdef class Heuristic:

    # Attributes:

    cdef shared_ptr[IHeuristic] heuristic_ptr


cdef class Precision(Heuristic):
    pass


cdef class Recall(Heuristic):
    pass


cdef class WRA(Heuristic):
    pass


cdef class HammingLoss(Heuristic):
    pass


cdef class FMeasure(Heuristic):
    pass


cdef class MEstimate(Heuristic):
    pass
