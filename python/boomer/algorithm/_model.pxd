cimport numpy as npc
ctypedef npc.int32_t int32
ctypedef npc.float32_t float32
ctypedef npc.float64_t float64


cdef class Body:

    cdef bint covers(self, float32[:] example)


cdef class EmptyBody(Body):

    cdef bint covers(self, float32[:] example)


cdef class ConjunctiveBody(Body):

    cdef readonly int32[::1] leq_feature_indices

    cdef readonly float32[::1] leq_thresholds

    cdef readonly int32[::1] gr_feature_indices

    cdef readonly float32[::1] gr_thresholds

    cdef bint covers(self, float32[:] example)


cdef class Head:

    cdef predict(self, float64[:] predictions)


cdef class FullHead(Head):

    cdef readonly float64[::1] scores

    cdef predict(self, float64[:] predictions)


cdef class PartialHead(Head):

    cdef readonly int32[::1] label_indices

    cdef readonly float64[::1] scores

    cdef predict(self, float64[:] predictions)


cdef class Rule:

    cdef readonly Body body

    cdef readonly Head head

    cpdef predict(self, float32[::1, :] x, float64[::1, :] predictions)
