"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides classes for making predictions using rule-based models.
"""
from boomer.algorithm._arrays cimport uint8, uint32, array_uint32, array_float32, c_matrix_uint8, c_matrix_float64
from boomer.algorithm.rules cimport Rule

import numpy as np


cdef class Aggregation:
    """
    Allows to aggregate the predictions provided by several rules.
    """

    def __cinit__(self, bint use_mask = False):
        """
        :param use_mask: True, if only one rule is allowed to predict per label, False otherwise
        """
        self.use_mask = use_mask

    cdef float64[:, ::1] predict(self, float32[:, ::1] x, intp num_labels, list rules):
        """
        Aggregates and returns the predictions provided by several rules.

        The feature matrix must be given as a dense C-contiguous array.

        :param x:           An array of dtype float, shape `(num_examples, num_features)`, representing the features of
                            the examples to predict for
        :param num_labels:  The total number of labels
        :param rules:       A list that contains the rules
        :return:            An array of dtype float, shape `(num_examples, num_labels)`, representing the predictions
                            for individual examples and labels
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
        Aggregates and returns the predictions provided by several rules.

        The feature matrix must be given in compressed sparse row (CSR) format.

        :param x_data:          An array of dtype float, shape `(num_non_zero_feature_values)`, representing the
                                non-zero feature values of the training examples
        :param x_row_indices:   An array of dtype int, shape `(num_examples + 1)`, representing the indices of the first
                                element in `x_data` and `x_col_indices` that corresponds to a certain examples. The
                                index at the last position is equal to `num_non_zero_feature_values`
        :param x_col_indices:   An array of dtype int, shape `(num_non_zero_feature_values)`, representing the
                                column-indices of the examples, the values in `x_data` correspond to
        :param num_features:    The total number of features
        :param num_labels:      The total number of labels
        :param rules:           A list that contains the rules
        :return:                An array of dtype float, shape `(num_examples, num_labels)`, representing the
                                predictions for individual examples and labels
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
    A base class for all classes that allow to make predictions based on rule-based models.
    """

    cpdef object predict(self, float32[:, ::1] x, intp num_labels, list rules):
        """
        Obtains and returns the predictions for given examples.

        The feature matrix must be given as a dense C-contiguous array.

        :param x:           An array of dtype float, shape `(num_examples, num_features)`, representing the features of
                            the examples to predict for
        :param num_labels:  The total number of labels
        :param rules:       A list that contains the rules
        :return:            A `np.ndarray` or a `scipy.sparse.matrix`, shape `(num_examples, num_labels)`, representing
                            the predictions for individual examples and labels
        """
        pass


    cpdef object predict_csr(self, float32[::1] x_data, intp[::1] x_row_indices, intp[::1] x_col_indices,
                             intp num_features, intp num_labels, list rules):
        """
        Obtains and returns the predictions for given examples.

        The feature matrix must be given in compressed sparse row (CSR) format.

        :param x_data:          An array of dtype float, shape `(num_non_zero_feature_values)`, representing the
                                non-zero feature values of the training examples
        :param x_row_indices:   An array of dtype int, shape `(num_examples + 1)`, representing the indices of the first
                                element in `x_data` and `x_col_indices` that corresponds to a certain examples. The
                                index at the last position is equal to `num_non_zero_feature_values`
        :param x_col_indices:   An array of dtype int, shape `(num_non_zero_feature_values)`, representing the
                                column-indices of the examples, the values in `x_data` correspond to
        :param num_features:    The total number of features
        :param num_labels:      The total number of labels
        :param rules:           A list that contains the rules
        :return:                A `np.ndarray` or a `scipy.sparse.matrix`, shape `(num_examples, num_labels)`,
                                representing the predictions for individual examples and labels
        """
        pass


cdef class DensePredictor(Predictor):
    """
    Allows to make predictions based on rule-based models that are stored in dense matrices.
    """

    def __cinit__(self, Aggregation aggregation, Transformation transformation):
        """
        :param aggregation:     The aggregation to be used to obtain raw predictions from the individual rules
        :param transformation:  The transformation to be applied to the raw predictions
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
    A base class for all transformation functions that may be applied to predictions.
    """

    cdef object transform_matrix(self, float64[:, ::1] m):
        """
        Applies the transformation function to a matrix.

        :param m:   An array of dtype float, shape `(num_rows, num_cols)`, the transformation function should be
                    applied to
        :return:    A `np.ndarray` or `scipy.sparse.matrix`, shape `(num_rows, num_cols)`, representing the result of
                    the transformation
        """
        pass


cdef class SignFunction(Transformation):
    """
    Transforms predictions according to the sign function (1 if x > 0, 0 otherwise).
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
