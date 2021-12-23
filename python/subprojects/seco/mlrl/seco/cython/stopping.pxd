from mlrl.common.cython._types cimport float64
from mlrl.common.cython.stopping cimport IStoppingCriterionFactory, StoppingCriterionFactory


cdef extern from "seco/stopping/stopping_criterion_coverage.hpp" nogil:

    cdef cppclass CoverageStoppingCriterionFactoryImpl"seco::CoverageStoppingCriterionFactory"(
            IStoppingCriterionFactory):

        # Constructors:

        CoverageStoppingCriterionFactoryImpl(float64 threshold) except +


cdef class CoverageStoppingCriterionFactory(StoppingCriterionFactory):
    pass
