from boomer.common._types cimport uint32

from libcpp.memory cimport shared_ptr


cdef extern from "cpp/stopping/stopping_criterion.hpp" nogil:

    cdef cppclass IStoppingCriterion:
        pass


cdef extern from "cpp/stopping/stopping_criterion_size.hpp" nogil:

    cdef cppclass SizeStoppingCriterionImpl"SizeStoppingCriterion"(IStoppingCriterion):

        # Constructors:

        SizeStoppingCriterionImpl(uint32 maxRules) except +


cdef extern from "cpp/stopping/stopping_criterion_time.hpp" nogil:

    cdef cppclass TimeStoppingCriterionImpl"TimeStoppingCriterion"(IStoppingCriterion):

        # Constructors:

        TimeStoppingCriterionImpl(uint32 timeLimit) except +


cdef class StoppingCriterion:

    # Attributes:

    cdef shared_ptr[IStoppingCriterion] stopping_criterion_ptr


cdef class SizeStoppingCriterion(StoppingCriterion):
    pass


cdef class TimeStoppingCriterion(StoppingCriterion):
    pass
