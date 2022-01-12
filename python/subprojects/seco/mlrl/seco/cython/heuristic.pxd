from mlrl.common.cython._types cimport float64


cdef extern from "seco/heuristics/heuristic_accuracy.hpp" namespace "seco" nogil:

    cdef cppclass AccuracyConfigImpl"seco::AccuracyConfig":
        pass


cdef extern from "seco/heuristics/heuristic_f_measure.hpp" namespace "seco" nogil:

    cdef cppclass FMeasureConfigImpl"seco::FMeasureConfig":

        # Functions:

        float64 getBeta() const

        FMeasureConfigImpl& setBeta(float64 beta) except +


cdef extern from "seco/heuristics/heuristic_laplace.hpp" namespace "seco" nogil:

    cdef cppclass LaplaceConfigImpl"seco::LaplaceConfig":
        pass


cdef extern from "seco/heuristics/heuristic_m_estimate.hpp" namespace "seco" nogil:

    cdef cppclass MEstimateConfigImpl"seco::MEstimateConfig":

        # Functions:

        float64 getM() const

        MEstimateConfigImpl& setM(float64 m) except +


cdef extern from "seco/heuristics/heuristic_precision.hpp" namespace "seco" nogil:

    cdef cppclass PrecisionConfigImpl"seco::PrecisionConfig":
        pass


cdef extern from "seco/heuristics/heuristic_recall.hpp" namespace "seco" nogil:

    cdef cppclass RecallConfigImpl"seco::RecallConfig":
        pass


cdef extern from "seco/heuristics/heuristic_wra.hpp" namespace "seco" nogil:

    cdef cppclass WraConfigImpl"seco::WraConfig":
        pass


cdef class AccuracyConfig:

    # Attributes:

    cdef AccuracyConfigImpl* config_ptr


cdef class FMeasureConfig:

    # Attributes:

    cdef FMeasureConfigImpl* config_ptr


cdef class LaplaceConfig:

    # Attributes:

    cdef LaplaceConfigImpl* config_ptr


cdef class MEstimateConfig:

    # Attributes:

    cdef MEstimateConfigImpl* config_ptr


cdef class PrecisionConfig:

    # Attributes:

    cdef PrecisionConfigImpl* config_ptr


cdef class RecallConfig:

    # Attributes:

    cdef RecallConfigImpl* config_ptr


cdef class WraConfig:

    # Attributes:

    cdef WraConfigImpl* config_ptr
