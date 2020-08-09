"""
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides Cython wrappers for C++ classes that implement different heuristics for assessing the quality of rules.
"""
from libcpp.memory cimport make_shared


cdef class Heuristic:
    """
    A wrapper for the abstract C++ class `AbstractHeuristic`.
    """
    pass


cdef class Precision(Heuristic):
    """
    A wrapper for the C++ class `PrecisionImpl`.
    """

    def __cinit__(self):
        self.heuristic_ptr = <shared_ptr[AbstractHeuristic]>make_shared[PrecisionImpl]()


cdef class Recall(Heuristic):
    """
    A wrapper for the C++ class `RecallImpl`.
    """

    def __cinit__(self):
        self.heuristic_ptr = <shared_ptr[AbstractHeuristic]>make_shared[RecallImpl]()


cdef class WRA(Heuristic):
    """
    A wrapper for the C++ class `WRAImpl`.
    """

    def __cinit__(self):
        self.heuristic_ptr = <shared_ptr[AbstractHeuristic]>make_shared[WRAImpl]()


cdef class HammingLoss(Heuristic):
    """
    A wrapper for the C++ class `HammingLossImpl`.
    """

    def __cinit__(self):
        self.heuristic_ptr = <shared_ptr[AbstractHeuristic]>make_shared[HammingLossImpl]()


cdef class FMeasure(Heuristic):
    """
    A wrapper for the C++ class `FMeasureImpl`.
    """

    def __cinit__(self, float64 beta = 0.5):
        """
        :param beta: The value of the beta-parameter. Must be at least 0
        """
        self.heuristic_ptr = <shared_ptr[AbstractHeuristic]>make_shared[FMeasureImpl](beta)


cdef class MEstimate(Heuristic):
    """
    A wrapper for the C++ class `MEstimateImpl`.
    """

    def __cinit__(self, float64 m = 22.466):
        """
        :param m: The value of the m-parameter. Must be at least 0
        """
        self.heuristic_ptr = <shared_ptr[AbstractHeuristic]>make_shared[MEstimateImpl](m)
