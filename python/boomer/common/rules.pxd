from boomer.common._arrays cimport uint8, uint32, intp, float32, float64


cdef class Body:

    # Functions:

    cdef bint covers(self, float32[::1] example)

    cdef bint covers_sparse(self, float32[::1] example_data, intp[::1] example_indices, float32[::1] tmp_array1,
                            uint32[::1] tmp_array2, uint32 n)


cdef class EmptyBody(Body):

    # Functions:

    cdef bint covers(self, float32[::1] example)

    cdef bint covers_sparse(self, float32[::1] example_data, intp[::1] example_indices, float32[::1] tmp_array1,
                            uint32[::1] tmp_array2, uint32 n)


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

    cdef bint covers(self, float32[::1] example)

    cdef bint covers_sparse(self, float32[::1] example_data, intp[::1] example_indices, float32[::1] tmp_array1,
                            uint32[::1] tmp_array2, uint32 n)


cdef class Head:

    # Functions:

    cdef void predict(self, float64[::1] predictions, uint8[::1] mask=*)


cdef class FullHead(Head):

    # Attributes:

    cdef readonly float64[::1] scores

    # Functions:

    cdef void predict(self, float64[::1] predictions, uint8[::1] mask=*)


cdef class PartialHead(Head):

    # Attributes:

    cdef readonly intp[::1] label_indices

    cdef readonly float64[::1] scores

    # Functions:

    cdef void predict(self, float64[::1] predictions, uint8[::1] mask=*)


cdef class Rule:

    # Attributes:

    cdef readonly Body body

    cdef readonly Head head

    # Functions:

    cpdef predict(self, float32[:, ::1] x, float64[:, ::1] predictions, uint8[:, ::1] mask=*)

    cpdef predict_csr(self, float32[::1] x_data, intp[::1] x_row_indices, intp[::1] x_col_indices, intp num_features,
                      float32[::1] tmp_array1, uint32[::1] tmp_array2, uint32 n, float64[:, ::1] predictions,
                      uint8[:, ::1] mask=*)
