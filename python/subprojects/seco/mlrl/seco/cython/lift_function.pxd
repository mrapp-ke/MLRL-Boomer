from mlrl.common.cython._types cimport uint32, float64


cdef extern from "seco/rule_evaluation/lift_function_peak.hpp" namespace "seco" nogil:

    cdef cppclass IPeakLiftFunctionConfig:

        # Functions:

        uint32 getNumLabels() const

        IPeakLiftFunctionConfig& setNumLabels(uint32 numLabels) except +

        uint32 getPeakLabel() const

        IPeakLiftFunctionConfig& setPeakLabel(uint32 peakLabel) except +

        float64 getMaxLift() const

        IPeakLiftFunctionConfig& setMaxLift(float64 maxLift) except +

        float64 getCurvature() const

        IPeakLiftFunctionConfig& setCurvature(float64 curvature) except +


cdef class PeakLiftFunctionConfig:

    # Attributes:

    cdef IPeakLiftFunctionConfig* config_ptr
