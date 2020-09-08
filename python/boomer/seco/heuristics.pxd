from boomer.common._arrays cimport float64

from libcpp.memory cimport shared_ptr


cdef extern from "cpp/heuristics.h" namespace "seco" nogil:

    cdef cppclass AbstractHeuristic:
        pass


    cdef cppclass PrecisionImpl(AbstractHeuristic):
        pass


    cdef cppclass RecallImpl(AbstractHeuristic):
        pass


    cdef cppclass WRAImpl(AbstractHeuristic):
        pass


    cdef cppclass HammingLossImpl(AbstractHeuristic):
        pass


    cdef cppclass FMeasureImpl(AbstractHeuristic):

        # Constructors:

        FMeasureImpl(float64 beta) except +


    cdef cppclass MEstimateImpl(AbstractHeuristic):

        # Constructors:

        MEstimateImpl(float64 m) except +


cdef class Heuristic:

    # Attributes:

    cdef shared_ptr[AbstractHeuristic] heuristic_ptr


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
