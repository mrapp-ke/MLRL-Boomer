from boomer.common._types cimport float64
from boomer.common.stopping_criteria cimport IStoppingCriterion, StoppingCriterion


cdef extern from "cpp/stopping/stopping_criterion_coverage.h" nogil:

    cdef cppclass CoverageStoppingCriterionImpl"seco::CoverageStoppingCriterion"(IStoppingCriterion):

        # Constructors:

        CoverageStoppingCriterionImpl(float64 threshold) except +


cdef class CoverageStoppingCriterion(StoppingCriterion):
    pass
