"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides classes for making predictions using rule-based models.
"""
from boomer.algorithm._arrays cimport uint32, float64, array_uint32, array_float32, c_matrix_float64
from boomer.algorithm.rules cimport Rule

import numpy as np


cdef class Predictor:
    """
    TODO
    """

    cpdef object predict(self, float32[:, ::1] x, intp num_labels, list rules):
        """
        TODO

        :param x:
        :param num_labels:
        :param rules:
        :return:
        """
        pass

    cpdef object predict_csr(self, float32[::1] x_data, intp[::1] x_row_indices, intp[::1] x_col_indices,
                             intp num_features, intp num_labels, list rules):
        """
        TODO

        :param x_data:
        :param x_row_indices:
        :param x_col_indices:
        :param num_features:
        :param num_labels:
        :param rules:
        :return:
        """
        pass


cdef class RawPredictor(Predictor):
    """
    TODO
    """

    cpdef object predict(self, float32[:, ::1] x, intp num_labels, list rules):
        cdef intp num_examples = x.shape[0]
        cdef float64[:, ::1] predictions = c_matrix_float64(num_examples, num_labels)
        predictions[:, :] = 0
        cdef Rule rule

        for rule in rules:
            rule.predict(x, predictions)

        return np.asarray(predictions)

    cpdef object predict_csr(self, float32[::1] x_data, intp[::1] x_row_indices, intp[::1] x_col_indices,
                             intp num_features, intp num_labels, list rules):
        cdef intp num_examples = x_row_indices.shape[0] - 1
        cdef float64[:, ::1] predictions = c_matrix_float64(num_examples, num_labels)
        predictions[:, :] = 0
        cdef float32[::1] tmp_array1 = array_float32(num_features)
        cdef uint32[::1] tmp_array2 = array_uint32(num_features)
        tmp_array2[:] = 0
        cdef uint32 n = 1
        cdef Rule rule

        for rule in rules:
            rule.predict_csr(x_data, x_row_indices, x_col_indices, num_features, tmp_array1, tmp_array2, n, predictions)
            n += 1

        return np.asarray(predictions)
