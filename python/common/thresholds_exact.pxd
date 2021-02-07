from common.thresholds cimport ThresholdsFactory, IThresholdsFactory


cdef extern from "cpp/thresholds/thresholds_exact.hpp" nogil:

    cdef cppclass ExactThresholdsFactoryImpl"ExactThresholdsFactory"(IThresholdsFactory):
        pass


cdef class ExactThresholdsFactory(ThresholdsFactory):
    pass
