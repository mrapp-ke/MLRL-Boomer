from common.cython._types cimport uint32
from common.cython._measures cimport IMeasure

from libcpp.memory cimport shared_ptr


cdef extern from "common/stopping/stopping_criterion.hpp" nogil:

    cdef cppclass IStoppingCriterion:
        pass


cdef extern from "common/stopping/stopping_criterion_size.hpp" nogil:

    cdef cppclass SizeStoppingCriterionImpl"SizeStoppingCriterion"(IStoppingCriterion):

        # Constructors:

        SizeStoppingCriterionImpl(uint32 maxRules) except +


cdef extern from "common/stopping/stopping_criterion_time.hpp" nogil:

    cdef cppclass TimeStoppingCriterionImpl"TimeStoppingCriterion"(IStoppingCriterion):

        # Constructors:

        TimeStoppingCriterionImpl(uint32 timeLimit) except +


cdef extern from "common/stopping/stopping_criterion_measure.hpp" nogil:

    cdef cppclass MeasureStoppingCriterionImpl"MeasureStoppingCriterion"(IStoppingCriterion):

        # Constructors:

        MeasureStoppingCriterionImpl(shared_ptr[IMeasure] measurePtr) except +


cdef class StoppingCriterion:

    # Attributes:

    cdef shared_ptr[IStoppingCriterion] stopping_criterion_ptr


cdef class SizeStoppingCriterion(StoppingCriterion):
    pass


cdef class TimeStoppingCriterion(StoppingCriterion):
    pass
