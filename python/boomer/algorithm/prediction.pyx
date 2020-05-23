"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides classes for making predictions using rule-based models.
"""
from boomer.algorithm._arrays cimport uint8, uint32, array_uint32, array_float32, c_matrix_uint8, c_matrix_float64
from boomer.algorithm.rules cimport Rule

import numpy as np


cdef class Aggregation:
    """
    TODO
    """

    def __cinit__(self, bint use_mask = False):
        """
        :param use_mask: True, if only one rule is allowed to predict per label, False otherwise
        """
        self.use_mask = use_mask

    cdef float64[:, ::1] predict(self, float32[:, ::1] x, intp num_labels, list rules):
        """
        TODO

        :param x:
        :param num_labels:
        :param rules:
        :return:
        """
        cdef intp num_examples = x.shape[0]
        cdef float64[:, ::1] predictions = c_matrix_float64(num_examples, num_labels)
        predictions[:, :] = 0
        cdef bint use_mask = self.use_mask
        cdef uint8[:, ::1] mask

        if use_mask:
            mask = c_matrix_uint8(num_examples, num_labels)
            mask[:, :] = True
        else:
            mask = None

        cdef Rule rule

        for rule in rules:
            rule.predict(x, predictions, mask)

        return predictions

    cdef float64[:, ::1] predict_csr(self, float32[::1] x_data, intp[::1] x_row_indices, intp[::1] x_col_indices,
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
        cdef intp num_examples = x_row_indices.shape[0] - 1
        cdef float64[:, ::1] predictions = c_matrix_float64(num_examples, num_labels)
        predictions[:, :] = 0
        cdef bint use_mask = self.use_mask
        cdef uint8[:, ::1] mask

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

        return predictions


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


cdef class DensePredictor(Predictor):
    """
    TODO
    """

    def __cinit__(self, Aggregation aggregation, Transformation transformation):
        """
        TODO
        :param aggregation:
        :param transformation:
        """
        self.aggregation = aggregation
        self.transformation = transformation

    cpdef object predict(self, float32[:, ::1] x, intp num_labels, list rules):
        cdef Aggregation aggregation = self.aggregation
        cdef float64[:, ::1] predictions = aggregation.predict(x, num_labels, rules)
        cdef Transformation transformation = self.transformation

        if transformation is not None:
            return transformation.transform_matrix(predictions)
        else:
            return np.asarray(predictions)


    cpdef object predict_csr(self, float32[::1] x_data, intp[::1] x_row_indices, intp[::1] x_col_indices,
                             intp num_features, intp num_labels, list rules):
        cdef Aggregation aggregation = self.aggregation
        cdef float64[:, ::1] predictions = aggregation.predict_csr(x_data, x_row_indices, x_col_indices, num_features,
                                                                   num_labels, rules)
        cdef Transformation transformation = self.transformation

        if transformation is not None:
            return transformation.transform_matrix(predictions)
        else:
            return np.asarray(predictions)



cdef class Transformation:
    """
    TODO
    """

    cdef object transform_matrix(self, float64[:, ::1] m):
        """
        TODO

        :param m:
        :return:
        """
        pass


cdef class SignFunction(Transformation):
    """
    TODO
    """

    cdef object transform_matrix(self, float64[:, ::1] m):
        cdef intp num_rows = m.shape[0]
        cdef intp num_cols = m.shape[1]
        cdef uint8[:, ::1] result = c_matrix_uint8(num_rows, num_cols)
        cdef intp r, c

        for r in range(num_rows):
            for c in range(num_cols):
                result[r, c] = m[r, c] > 0

        return np.asarray(result)
