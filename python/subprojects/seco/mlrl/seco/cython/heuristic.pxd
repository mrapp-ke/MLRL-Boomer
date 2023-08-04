from mlrl.common.cython._types cimport float64


cdef extern from "mlrl/seco/heuristics/heuristic_f_measure.hpp" namespace "seco" nogil:

    cdef cppclass IFMeasureConfig:

        # Functions:

        float64 getBeta() const

        IFMeasureConfig& setBeta(float64 beta) except +


cdef extern from "mlrl/seco/heuristics/heuristic_m_estimate.hpp" namespace "seco" nogil:

    cdef cppclass IMEstimateConfig:

        # Functions:

        float64 getM() const

        IMEstimateConfig& setM(float64 m) except +


cdef class FMeasureConfig:

    # Attributes:

    cdef IFMeasureConfig* config_ptr


cdef class MEstimateConfig:

    # Attributes:

    cdef IMEstimateConfig* config_ptr
