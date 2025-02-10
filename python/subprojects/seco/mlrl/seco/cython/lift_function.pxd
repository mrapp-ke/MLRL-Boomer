from mlrl.common.cython._types cimport float32, uint32


cdef extern from "mlrl/seco/lift_functions/lift_function_peak.hpp" namespace "seco" nogil:

    cdef cppclass IPeakLiftFunctionConfig:

        # Functions:

        uint32 getPeakLabel() const

        IPeakLiftFunctionConfig& setPeakLabel(uint32 peakLabel) except +

        float32 getMaxLift() const

        IPeakLiftFunctionConfig& setMaxLift(float32 maxLift) except +

        float32 getCurvature() const

        IPeakLiftFunctionConfig& setCurvature(float32 curvature) except +


cdef extern from "mlrl/seco/lift_functions/lift_function_kln.hpp" namespace "seco" nogil:

    cdef cppclass IKlnLiftFunctionConfig:

        # Functions:

        float32 getK() const

        IKlnLiftFunctionConfig& setK(float32 k) except +


cdef class PeakLiftFunctionConfig:

    # Attributes:

    cdef IPeakLiftFunctionConfig* config_ptr


cdef class KlnLiftFunctionConfig:

    # Attributes:

    cdef IKlnLiftFunctionConfig* config_ptr
