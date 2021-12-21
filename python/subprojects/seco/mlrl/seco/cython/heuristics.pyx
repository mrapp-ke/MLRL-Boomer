"""
@author Jakob Steeg (jakob.steeg@gmail.com)
@author Michael Rapp (michael.rapp.ml@gmail.com)
@author Andreas Seidl Fernandez (aseidlfernandez@gmail.com)
"""
from libcpp.memory cimport make_unique


cdef class HeuristicFactory:
    """
    A wrapper for the pure virtual C++ class `IHeuristicFactory`.
    """
    pass


cdef class AccuracyFactory(HeuristicFactory):
    """
    A wrapper for the C++ class `AccuracyFactory`.
    """

    def __cinit__(self):
        self.heuristic_factory_ptr = <unique_ptr[IHeuristicFactory]>make_unique[AccuracyFactoryImpl]()


cdef class PrecisionFactory(HeuristicFactory):
    """
    A wrapper for the C++ class `PrecisionFactory`.
    """

    def __cinit__(self):
        self.heuristic_factory_ptr = <unique_ptr[IHeuristicFactory]>make_unique[PrecisionFactoryImpl]()


cdef class RecallFactory(HeuristicFactory):
    """
    A wrapper for the C++ class `RecallFactory`.
    """

    def __cinit__(self):
        self.heuristic_factory_ptr = <unique_ptr[IHeuristicFactory]>make_unique[RecallFactoryImpl]()


cdef class LaplaceFactory(HeuristicFactory):
    """
    A wrapper for the C++ class 'LaplaceFactory'.
    """

    def __cinit__(self):
        self.heuristic_factory_ptr = <unique_ptr[IHeuristicFactory]>make_unique[LaplaceFactoryImpl]()


cdef class WraFactory(HeuristicFactory):
    """
    A wrapper for the C++ class `WraFactory`.
    """

    def __cinit__(self):
        self.heuristic_factory_ptr = <unique_ptr[IHeuristicFactory]>make_unique[WraFactoryImpl]()


cdef class FMeasureFactory(HeuristicFactory):
    """
    A wrapper for the C++ class `FMeasureFactory`.
    """

    def __cinit__(self, float64 beta):
        """
        :param beta: The value of the beta-parameter. Must be at least 0
        """
        self.heuristic_factory_ptr = <unique_ptr[IHeuristicFactory]>make_unique[FMeasureFactoryImpl](beta)


cdef class MEstimateFactory(HeuristicFactory):
    """
    A wrapper for the C++ class `MEstimateFactory`.
    """

    def __cinit__(self, float64 m):
        """
        :param m: The value of the m-parameter. Must be at least 0
        """
        self.heuristic_factory_ptr = <unique_ptr[IHeuristicFactory]>make_unique[MEstimateFactoryImpl](m)
