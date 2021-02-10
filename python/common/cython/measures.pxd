from common.cython._measures cimport IMeasure

from libcpp.memory cimport shared_ptr


cdef class Measure:

    # Functions:

    cdef shared_ptr[IMeasure] get_measure_ptr(self)
