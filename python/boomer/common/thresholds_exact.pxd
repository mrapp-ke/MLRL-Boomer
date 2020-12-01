from boomer.common.thresholds cimport ThresholdsFactory, IThresholdsFactory


cdef extern from "cpp/thresholds/thresholds_exact.h" nogil:

    cdef cppclass ExactThresholdsFactoryImpl"ExactThresholdsFactory"(IThresholdsFactory):
        pass


cdef class ExactThresholdsFactory(ThresholdsFactory):
    pass
