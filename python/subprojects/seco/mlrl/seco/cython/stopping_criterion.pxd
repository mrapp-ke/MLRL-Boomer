from mlrl.common.cython._types cimport float64


cdef extern from "seco/stopping/stopping_criterion_coverage.hpp" namespace "seco" nogil:

    cdef cppclass CoverageStoppingCriterionConfigImpl"seco::CoverageStoppingCriterionConfig":

        # Functions:

        float64 getThreshold() const

        CoverageStoppingCriterionConfigImpl& setThreshold(float64 threshold) except +


cdef class CoverageStoppingCriterionConfig:

    # Attributes:

    cdef CoverageStoppingCriterionConfigImpl* config_ptr
