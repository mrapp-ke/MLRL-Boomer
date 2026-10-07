from mlrl.common.cython._types cimport float32


cdef extern from "mlrl/seco/stopping/stopping_criterion_coverage.hpp" namespace "seco" nogil:

    cdef cppclass ICoverageStoppingCriterionConfig:

        # Functions:

        float32 getMinCoverage() const

        ICoverageStoppingCriterionConfig& setMinCoverage(float32 minCoverage) except +


cdef class CoverageStoppingCriterionConfig:

    # Attributes:

    cdef ICoverageStoppingCriterionConfig* config_ptr
