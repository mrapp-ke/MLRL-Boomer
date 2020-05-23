from boomer.algorithm._arrays cimport intp, float32, float64


cdef class Predictor:

    # Functions:

    cpdef object predict(self, float32[:, ::1] x, intp num_labels, list rules)

    cpdef object predict_csr(self, float32[::1] x_data, intp[::1] x_row_indices, intp[::1] x_col_indices,
                             intp num_features, intp num_labels, list rules)


cdef class RawPredictor(Predictor):

    # Arguments:

    cdef readonly bint use_mask

    # Functions:

    cpdef object predict(self, float32[:, ::1] x, intp num_labels, list rules)

    cpdef object predict_csr(self, float32[::1] x_data, intp[::1] x_row_indices, intp[::1] x_col_indices,
                             intp num_features, intp num_labels, list rules)


cdef class Converter(Predictor):

    # Arguments:

    cdef readonly RawPredictor predictor

    # Functions:

    cpdef object predict(self, float32[:, ::1] x, intp num_labels, list rules)

    cpdef object predict_csr(self, float32[::1] x_data, intp[::1] x_row_indices, intp[::1] x_col_indices,
                             intp num_features, intp num_labels, list rules)

    cdef object _convert(self, float64[:, ::1] raw_predictions)


cdef class Sign(Converter):

    # Functions:

    cpdef object predict(self, float32[:, ::1] x, intp num_labels, list rules)

    cpdef object predict_csr(self, float32[::1] x_data, intp[::1] x_row_indices, intp[::1] x_col_indices,
                             intp num_features, intp num_labels, list rules)

    cdef object _convert(self, float64[:, ::1] raw_predictions)
