from mlrl.common.cython._types cimport uint32, float32


cdef extern from "boosting/rule_evaluation/head_type_partial_fixed.hpp" namespace "boosting" nogil:

    cdef cppclass IFixedPartialHeadConfig:

        # Functions:

        float32 getLabelRatio() const

        IFixedPartialHeadConfig& setLabelRatio(float32 labelRatio) except +

        uint32 getMinLabels() const

        IFixedPartialHeadConfig& setMinLabels(uint32 minLabels) except +

        uint32 getMaxLabels() const

        IFixedPartialHeadConfig& setMaxLabels(uint32 maxLabels) except +


cdef class FixedPartialHeadConfig:

    # Attributes:

    cdef IFixedPartialHeadConfig* config_ptr
