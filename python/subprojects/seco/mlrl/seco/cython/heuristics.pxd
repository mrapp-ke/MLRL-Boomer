from mlrl.common.cython._types cimport float64

from libcpp.memory cimport unique_ptr


cdef extern from "seco/heuristics/heuristic.hpp" namespace "seco" nogil:

    cdef cppclass IHeuristicFactory:
        pass


cdef extern from "seco/heuristics/heuristic_accuracy.hpp" namespace "seco" nogil:

    cdef cppclass AccuracyFactoryImpl"seco::AccuracyFactory"(IHeuristicFactory):
        pass


cdef extern from "seco/heuristics/heuristic_precision.hpp" namespace "seco" nogil:

    cdef cppclass PrecisionFactoryImpl"seco::PrecisionFactory"(IHeuristicFactory):
        pass


cdef extern from "seco/heuristics/heuristic_recall.hpp" namespace "seco" nogil:

    cdef cppclass RecallFactoryImpl"seco::RecallFactory"(IHeuristicFactory):
        pass


cdef extern from "seco/heuristics/heuristic_laplace.hpp" namespace "seco" nogil:

    cdef cppclass LaplaceFactoryImpl"seco::LaplaceFactory"(IHeuristicFactory):
        pass


cdef extern from "seco/heuristics/heuristic_wra.hpp" namespace "seco" nogil:

    cdef cppclass WraFactoryImpl"seco::WraFactory"(IHeuristicFactory):
        pass


cdef extern from "seco/heuristics/heuristic_f_measure.hpp" namespace "seco" nogil:

    cdef cppclass FMeasureFactoryImpl"seco::FMeasureFactory"(IHeuristicFactory):

        # Constructors:

        FMeasureFactoryImpl(float64 beta) except +


cdef extern from "seco/heuristics/heuristic_m_estimate.hpp" namespace "seco" nogil:

    cdef cppclass MEstimateFactoryImpl"seco::MEstimateFactory"(IHeuristicFactory):

        # Constructors:

        MEstimateFactoryImpl(float64 m) except +


cdef class HeuristicFactory:

    # Attributes:

    cdef unique_ptr[IHeuristicFactory] heuristic_factory_ptr


cdef class AccuracyFactory(HeuristicFactory):
    pass


cdef class PrecisionFactory(HeuristicFactory):
    pass


cdef class RecallFactory(HeuristicFactory):
    pass


cdef class LaplaceFactory(HeuristicFactory):
    pass


cdef class WraFactory(HeuristicFactory):
    pass


cdef class FMeasureFactory(HeuristicFactory):
    pass


cdef class MEstimateFactory(HeuristicFactory):
    pass
