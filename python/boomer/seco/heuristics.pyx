"""
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides Cython wrappers for C++ classes that implement different heuristics for assessing the quality of rules.
"""


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
        self.heuristic = new PrecisionImpl()

    def __dealloc(self):
        del self.heuristic


cdef class Recall(Heuristic):
    """
    A wrapper for the C++ class `RecallImpl`.
    """

    def __cinit__(self):
        self.heuristic = new RecallImpl()

    def __dealloc(self):
        del self.heuristic


cdef class WRA(Heuristic):
    """
    A wrapper for the C++ class `WRAImpl`.
    """

    def __cinit__(self):
        self.heuristic = new WRAImpl()

    def __dealloc(self):
        del self.heuristic


cdef class HammingLoss(Heuristic):
    """
    A wrapper for the C++ class `HammingLossImpl`.
    """

    def __cinit__(self):
        self.heuristic = new HammingLossImpl()

    def __dealloc(self):
        del self.heuristic


cdef class FMeasure(Heuristic):
    """
    A wrapper for the C++ class `FMeasureImpl`.
    """

    def __cinit__(self, float64 beta = 0.5):
        """
        :param beta: The value of the beta-parameter. Must be at least 0
        """
        self.heuristic = new FMeasureImpl(beta)

    def __dealloc(self):
        del self.heuristic


cdef class MEstimate(Heuristic):
    """
    A wrapper for the C++ class `MEstimateImpl`.
    """

    def __cinit__(self, float64 m = 22.466):
        """
        :param m: The value of the m-parameter. Must be at least 0
        """
        self.heuristic = new MEstimateImpl(m)

    def __dealloc__(self):
        del self.heuristic
