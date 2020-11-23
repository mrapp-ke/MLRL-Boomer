from boomer.common._types cimport float64

from libcpp.memory cimport shared_ptr


cdef extern from "cpp/heuristics/heuristic.h" namespace "seco" nogil:

    cdef cppclass IHeuristic:
        pass


cdef extern from "cpp/heuristics/heuristic_precision.h" namespace "seco" nogil:

    cdef cppclass PrecisionImpl"seco::Precision"(IHeuristic):
        pass


cdef extern from "cpp/heuristics/heuristic_recall.h" namespace "seco" nogil:

    cdef cppclass RecallImpl"seco::Recall"(IHeuristic):
        pass


cdef extern from "cpp/heuristics/heuristic_wra.h" namespace "seco" nogil:

    cdef cppclass WRAImpl"seco::WRA"(IHeuristic):
        pass


cdef extern from "cpp/heuristics/heuristic_hamming_loss.h" namespace "seco" nogil:

    cdef cppclass HammingLossImpl"seco::HammingLoss"(IHeuristic):
        pass


cdef extern from "cpp/heuristics/heuristic_f_measure.h" namespace "seco" nogil:

    cdef cppclass FMeasureImpl"seco::FMeasure"(IHeuristic):

        # Constructors:

        FMeasureImpl(float64 beta) except +


cdef extern from "cpp/heuristics/heuristic_m_estimate.h" namespace "seco" nogil:

    cdef cppclass MEstimateImpl"seco::MEstimate"(IHeuristic):

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
