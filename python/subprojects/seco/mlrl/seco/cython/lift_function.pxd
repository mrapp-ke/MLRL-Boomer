from mlrl.common.cython._types cimport float64, uint32


cdef extern from "seco/lift_functions/lift_function_peak.hpp" namespace "seco" nogil:

    cdef cppclass IPeakLiftFunctionConfig:

        # Functions:

        uint32 getPeakLabel() const

        IPeakLiftFunctionConfig& setPeakLabel(uint32 peakLabel) except +

        float64 getMaxLift() const

        IPeakLiftFunctionConfig& setMaxLift(float64 maxLift) except +

        float64 getCurvature() const

        IPeakLiftFunctionConfig& setCurvature(float64 curvature) except +


cdef extern from "seco/lift_functions/lift_function_kln.hpp" namespace "seco" nogil:

    cdef cppclass IKlnLiftFunctionConfig:

        # Functions:

        float64 getK() const

        IKlnLiftFunctionConfig& setK(float64 k) except +


cdef class PeakLiftFunctionConfig:

    # Attributes:

    cdef IPeakLiftFunctionConfig* config_ptr


cdef class KlnLiftFunctionConfig:

    # Attributes:

    cdef IKlnLiftFunctionConfig* config_ptr
