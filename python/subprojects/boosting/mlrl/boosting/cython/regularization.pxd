from mlrl.common.cython._types cimport float32


cdef extern from "mlrl/boosting/rule_evaluation/regularization_manual.hpp" namespace "boosting" nogil:

    cdef cppclass IManualRegularizationConfig:

        # Functions:

        float32 getRegularizationWeight() const

        IManualRegularizationConfig& setRegularizationWeight(float32 regularizationWeight) except +


cdef class ManualRegularizationConfig:

    # Attributes:

    cdef IManualRegularizationConfig* config_ptr
