from mlrl.common.cython._types cimport float32


cdef extern from "mlrl/boosting/post_processing/shrinkage_constant.hpp" namespace "boosting" nogil:

    cdef cppclass IConstantShrinkageConfig:

        # Functions:

        float32 getShrinkage() const

        IConstantShrinkageConfig& setShrinkage(float32 shrinkage) except +


cdef class ConstantShrinkageConfig:

    # Attributes:

    cdef IConstantShrinkageConfig* config_ptr
