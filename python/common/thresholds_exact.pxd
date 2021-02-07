from common.thresholds cimport ThresholdsFactory, IThresholdsFactory


cdef extern from "common/thresholds/thresholds_exact.hpp" nogil:

    cdef cppclass ExactThresholdsFactoryImpl"ExactThresholdsFactory"(IThresholdsFactory):
        pass


cdef class ExactThresholdsFactory(ThresholdsFactory):
    pass
