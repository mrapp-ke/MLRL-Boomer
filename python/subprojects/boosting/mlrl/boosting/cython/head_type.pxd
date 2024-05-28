from mlrl.common.cython._types cimport float32, uint32


cdef extern from "mlrl/boosting/rule_evaluation/head_type_partial_fixed.hpp" namespace "boosting" nogil:

    cdef cppclass IFixedPartialHeadConfig:

        # Functions:

        float32 getOutputRatio() const

        IFixedPartialHeadConfig& setOutputRatio(float32 outputRatio) except +

        uint32 getMinOutputs() const

        IFixedPartialHeadConfig& setMinOutputs(uint32 minOutputs) except +

        uint32 getMaxOutputs() const

        IFixedPartialHeadConfig& setMaxOutputs(uint32 maxOutputs) except +


cdef extern from "mlrl/boosting/rule_evaluation/head_type_partial_dynamic.hpp" namespace "boosting" nogil:

    cdef cppclass IDynamicPartialHeadConfig:

        # Functions:

        float32 getThreshold() const

        IDynamicPartialHeadConfig& setThreshold(float32 threshold) except +

        float32 getExponent() const

        IDynamicPartialHeadConfig& setExponent(float32 exponent) except +


cdef class FixedPartialHeadConfig:

    # Attributes:

    cdef IFixedPartialHeadConfig* config_ptr


cdef class DynamicPartialHeadConfig:

    # Attributes:

    cdef IDynamicPartialHeadConfig* config_ptr
