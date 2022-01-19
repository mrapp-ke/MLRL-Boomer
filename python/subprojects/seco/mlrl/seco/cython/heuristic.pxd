from mlrl.common.cython._types cimport float64


cdef extern from "seco/heuristics/heuristic_accuracy.hpp" namespace "seco" nogil:

    cdef cppclass IAccuracyConfig:
        pass


cdef extern from "seco/heuristics/heuristic_f_measure.hpp" namespace "seco" nogil:

    cdef cppclass IFMeasureConfig:

        # Functions:

        float64 getBeta() const

        IFMeasureConfig& setBeta(float64 beta) except +


cdef extern from "seco/heuristics/heuristic_laplace.hpp" namespace "seco" nogil:

    cdef cppclass ILaplaceConfig:
        pass


cdef extern from "seco/heuristics/heuristic_m_estimate.hpp" namespace "seco" nogil:

    cdef cppclass IMEstimateConfig:

        # Functions:

        float64 getM() const

        IMEstimateConfig& setM(float64 m) except +


cdef extern from "seco/heuristics/heuristic_precision.hpp" namespace "seco" nogil:

    cdef cppclass IPrecisionConfig:
        pass


cdef extern from "seco/heuristics/heuristic_recall.hpp" namespace "seco" nogil:

    cdef cppclass IRecallConfig:
        pass


cdef extern from "seco/heuristics/heuristic_wra.hpp" namespace "seco" nogil:

    cdef cppclass IWraConfig:
        pass


cdef class AccuracyConfig:

    # Attributes:

    cdef IAccuracyConfig* config_ptr


cdef class FMeasureConfig:

    # Attributes:

    cdef IFMeasureConfig* config_ptr


cdef class LaplaceConfig:

    # Attributes:

    cdef ILaplaceConfig* config_ptr


cdef class MEstimateConfig:

    # Attributes:

    cdef IMEstimateConfig* config_ptr


cdef class PrecisionConfig:

    # Attributes:

    cdef IPrecisionConfig* config_ptr


cdef class RecallConfig:

    # Attributes:

    cdef IRecallConfig* config_ptr


cdef class WraConfig:

    # Attributes:

    cdef IWraConfig* config_ptr
