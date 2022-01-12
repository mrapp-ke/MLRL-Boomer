from mlrl.common.cython._types cimport uint32, float64


cdef extern from "seco/rule_evaluation/lift_function_peak.hpp" namespace "seco" nogil:

    cdef cppclass PeakLiftFunctionConfigImpl"seco::PeakLiftFunctionConfig":

        # Functions:

        uint32 getNumLabels() const

        PeakLiftFunctionConfigImpl& setNumLabels(uint32 numLabels) except +

        uint32 getPeakLabel() const

        PeakLiftFunctionConfigImpl& setPeakLabel(uint32 peakLabel) except +

        float64 getMaxLift() const

        PeakLiftFunctionConfigImpl& setMaxLift(float64 maxLift) except +

        float64 getCurvature() const

        PeakLiftFunctionConfigImpl& setCurvature(float64 curvature) except +


cdef class PeakLiftFunctionConfig:

    # Attributes:

    cdef PeakLiftFunctionConfigImpl* config_ptr
