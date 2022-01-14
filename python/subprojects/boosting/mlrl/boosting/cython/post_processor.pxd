from mlrl.common.cython._types cimport float64


cdef extern from "boosting/post_processing/shrinkage_constant.hpp" namespace "boosting" nogil:

    cdef cppclass ConstantShrinkageConfigImpl"boosting::ConstantShrinkageConfig":

        # Functions:

        float64 getShrinkage() const

        ConstantShrinkageConfigImpl& setShrinkage(float64 shrinkage) except +


cdef class ConstantShrinkageConfig:

    # Attributes:

    cdef ConstantShrinkageConfigImpl* config_ptr
