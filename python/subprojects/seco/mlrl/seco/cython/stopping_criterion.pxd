from mlrl.common.cython._types cimport float64


cdef extern from "mlrl/seco/stopping/stopping_criterion_coverage.hpp" namespace "seco" nogil:

    cdef cppclass ICoverageStoppingCriterionConfig:

        # Functions:

        float64 getThreshold() const

        ICoverageStoppingCriterionConfig& setThreshold(float64 threshold) except +


cdef class CoverageStoppingCriterionConfig:

    # Attributes:

    cdef ICoverageStoppingCriterionConfig* config_ptr
