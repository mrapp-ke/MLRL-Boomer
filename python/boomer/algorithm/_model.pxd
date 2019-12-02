cimport numpy as npc
ctypedef Py_ssize_t intp
ctypedef npc.uint8_t uint8
ctypedef npc.uint32_t uint32
ctypedef npc.float32_t float32
ctypedef npc.float64_t float64


# A struct that represents a condition of a rule. It consists of the index of the feature that is used by the condition,
# whether it uses the <= or > operator, as well as a threshold.
cdef struct s_condition:
    intp feature_index
    bint leq
    float32 threshold


cdef inline s_condition make_condition(intp feature_index, bint leq, float32 threshold):
    """
    Creates and returns a new condition.

    :param feature_index:   The index of the feature that is used by the condition
    :param leq:             Whether the <= operator, or the > operator is used by the condition
    :param threshold:       The threshold that is used by the condition
    """
    cdef s_condition condition
    condition.feature_index = feature_index
    condition.leq = leq
    condition.threshold = threshold
    return condition


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

    # Functions:

    cdef bint covers(self, float32[:] example)


cdef class Head:

    # Functions:

    cdef predict(self, float64[:] predictions)


cdef class FullHead(Head):

    # Attributes:

    cdef readonly float64[::1] scores

    # Functions:

    cdef predict(self, float64[:] predictions)


cdef class PartialHead(Head):

    # Attributes:

    cdef readonly intp[::1] label_indices

    cdef readonly float64[::1] scores

    # Functions:

    cdef predict(self, float64[:] predictions)


cdef class Rule:

    # Attributes:

    cdef readonly Body body

    cdef readonly Head head

    # Functions:

    cpdef predict(self, float32[::1, :] x, float64[::1, :] predictions)
