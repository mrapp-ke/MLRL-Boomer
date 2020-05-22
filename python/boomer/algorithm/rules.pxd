from boomer.algorithm._arrays cimport intp, float32, float64


cdef class Body:

    # Functions:

    cdef bint covers(self, float32[:] example)


cdef class EmptyBody(Body):

    # Functions:

    cdef bint covers(self, float32[:] example)


cdef class ConjunctiveBody(Body):

    # Attributes:

    cdef readonly intp[::1] leq_feature_indices

    cdef readonly float32[::1] leq_thresholds

    cdef readonly intp[::1] gr_feature_indices

    cdef readonly float32[::1] gr_thresholds

    cdef readonly intp[::1] eq_feature_indices

    cdef readonly float32[::1] eq_thresholds

    cdef readonly intp[::1] neq_feature_indices

    cdef readonly float32[::1] neq_thresholds

    # Functions:

    cdef bint covers(self, float32[:] example)


cdef class Head:

    # Functions:

    cdef void predict(self, float64[::1] predictions, intp[::1] predicted=*)


cdef class FullHead(Head):

    # Attributes:

    cdef readonly float64[::1] scores

    # Functions:

    cdef void predict(self, float64[::1] predictions, intp[::1] predicted=*)


cdef class PartialHead(Head):

    # Attributes:

    cdef readonly intp[::1] label_indices

    cdef readonly float64[::1] scores

    # Functions:

    cdef void predict(self, float64[::1] predictions, intp[::1] predicted=*)


cdef class Rule:

    # Attributes:

    cdef readonly Body body

    cdef readonly Head head

    # Functions:

    cpdef predict(self, float32[::1, :] x, float64[:, ::1] predictions, intp[:, ::1] predicted=*)
