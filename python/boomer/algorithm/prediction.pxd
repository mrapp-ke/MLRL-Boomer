from boomer.algorithm._arrays cimport intp, float32, float64


cdef class Aggregation:

    # Attributes:

    cdef readonly bint use_mask

    # Functions:

    cdef float64[:, ::1] predict(self, float32[:, ::1] x, intp num_labels, list rules)

    cdef float64[:, ::1] predict_csr(self, float32[::1] x_data, intp[::1] x_row_indices, intp[::1] x_col_indices,
                                     intp num_features, intp num_labels, list rules)


cdef class Predictor:

    # Functions:

    cpdef object predict(self, float32[:, ::1] x, intp num_labels, list rules)

    cpdef object predict_csr(self, float32[::1] x_data, intp[::1] x_row_indices, intp[::1] x_col_indices,
                             intp num_features, intp num_labels, list rules)


cdef class DensePredictor(Predictor):

    # Attributes:

    cdef readonly Aggregation aggregation

    cdef readonly Transformation transformation

    # Functions:

    cpdef object predict(self, float32[:, ::1] x, intp num_labels, list rules)

    cpdef object predict_csr(self, float32[::1] x_data, intp[::1] x_row_indices, intp[::1] x_col_indices,
                             intp num_features, intp num_labels, list rules)

cdef class Transformation:

    # Functions:

    cdef object transform_matrix(self, float64[:, ::1] m)


cdef class SignFunction(Transformation):

    # Functions:

    cdef object transform_matrix(self, float64[:, ::1] m)
