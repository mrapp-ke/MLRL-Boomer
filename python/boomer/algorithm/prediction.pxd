from boomer.algorithm._arrays cimport intp, float32


cdef class Predictor:

    cpdef object predict(self, float32[:, ::1] x, intp num_labels, list rules)

    cpdef object predict_csr(self, float32[::1] x_data, intp[::1] x_row_indices, intp[::1] x_col_indices,
                             intp num_features, intp num_labels, list rules)


cdef class RawPredictor(Predictor):

    cpdef object predict(self, float32[:, ::1] x, intp num_labels, list rules)

    cpdef object predict_csr(self, float32[::1] x_data, intp[::1] x_row_indices, intp[::1] x_col_indices,
                             intp num_features, intp num_labels, list rules)
