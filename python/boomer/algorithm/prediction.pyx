"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides classes for making predictions using rule-based models.
"""
from boomer.algorithm._arrays cimport uint8, uint32, array_uint32, array_float32, c_matrix_uint8, c_matrix_float64
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

    def __cinit__(self, bint use_mask = False):
        """
        :param use_mask: True, if only one rule is allowed to predict per label, False otherwise
        """
        self.use_mask = use_mask

    cpdef object predict(self, float32[:, ::1] x, intp num_labels, list rules):
        cdef intp num_examples = x.shape[0]
        cdef float64[:, ::1] predictions = c_matrix_float64(num_examples, num_labels)
        predictions[:, :] = 0
        cdef bint use_mask = self.use_mask

        if use_mask:
            mask = c_matrix_uint8(num_examples, num_labels)
            mask[:, :] = True
        else:
            mask = None

        cdef Rule rule

        for rule in rules:
            rule.predict(x, predictions, mask)

        return np.asarray(predictions)

    cpdef object predict_csr(self, float32[::1] x_data, intp[::1] x_row_indices, intp[::1] x_col_indices,
                             intp num_features, intp num_labels, list rules):
        cdef intp num_examples = x_row_indices.shape[0] - 1
        cdef float64[:, ::1] predictions = c_matrix_float64(num_examples, num_labels)
        predictions[:, :] = 0
        cdef bint use_mask = self.use_mask

        if use_mask:
            mask = c_matrix_uint8(num_examples, num_labels)
            mask[:, :] = True
        else:
            mask = None

        cdef float32[::1] tmp_array1 = array_float32(num_features)
        cdef uint32[::1] tmp_array2 = array_uint32(num_features)
        tmp_array2[:] = 0
        cdef uint32 n = 1
        cdef Rule rule

        for rule in rules:
            rule.predict_csr(x_data, x_row_indices, x_col_indices, num_features, tmp_array1, tmp_array2, n, predictions,
                             mask)
            n += 1

        return np.asarray(predictions)


cdef class Converter(Predictor):
    """
    TODO
    """

    cpdef object predict(self, float32[:, ::1] x, intp num_labels, list rules):
        cdef RawPredictor predictor = self.predictor
        cdef float64[:, ::1] raw_predictions = predictor.predict(x, num_labels, rules)
        return self._convert(raw_predictions)

    cpdef object predict_csr(self, float32[::1] x_data, intp[::1] x_row_indices, intp[::1] x_col_indices,
                             intp num_features, intp num_labels, list rules):
        cdef RawPredictor predictor = self.predictor
        cdef float64[:, ::1] raw_predictions = predictor.predict_csr(x_data, x_row_indices, x_col_indices, num_features,
                                                                     num_labels, rules)
        return self._convert(raw_predictions)

    cdef object _convert(self, float64[:, ::1] raw_predictions):
        """
        TODO

        :param raw_predictions:
        :return:
        """
        pass


cdef class Sign(Converter):
    """
    TODO
    """

    def __cinit__(self, RawPredictor predictor):
        """
        :param predictor: The predictor that should be used to obtain raw predictions
        """
        self.predictor = predictor

    cdef object _convert(self, float64[:, ::1] raw_predictions):
        cdef intp num_examples = raw_predictions.shape[0]
        cdef intp num_labels = raw_predictions.shape[1]
        cdef uint8[:, ::1] converted_predictions = c_matrix_uint8(num_examples, num_labels)
        cdef intp r, c

        for r in range(num_examples):
            for c in range(num_labels):
                converted_predictions[r, c] = (raw_predictions[r, c] > 0)

        return np.asarray(converted_predictions)
