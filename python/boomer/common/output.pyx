"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides classes for making predictions using rule-based models.
"""


cdef class Predictor:
    """
    A base class for all classes that allow to make predictions based on rule-based models.
    """

    cpdef object predict(self, float32[:, ::1] array, RuleModel model):
        """
        Obtains and returns the predictions for given examples.

        The feature matrix must be given as a dense C-contiguous array.

        :param array:   An array of type `float32`, shape `(num_examples, num_features)`, that stores the feature values
                        of the examples to predict for
        :param model:   The model to be used for making predictions
        :return:        A `np.ndarray` or a `scipy.sparse` matrix, shape `(num_examples, num_labels)`, that stores the
                        predictions for individual examples and labels
        """
        pass

    cpdef object predict_csr(self, float32[::1] data, uint32[::1] row_indices, uint32[::1] col_indices,
                             uint32 num_features, RuleModel model):
        """
        Obtains and returns the predictions for given examples.

        The feature matrix must be given in compressed sparse row (CSR) format.

        :param data:            An array of type `float32`, shape `(num_non_zero_values)`, that stores all non-zero
                                feature values of the examples to predict for
        :param row_indices:     An array of type `uint32`, shape `(num_examples + 1)`, that stores the indices of the
                                first element in `data` and `col_indices` that corresponds to a certain example. The
                                index at the last position is equal to `num_non_zero_values`
        :param col_indices:     An array of type `uint32`, shape `(num_non_zero_values)`, that stores the
                                column-indices, the values in `data` correspond to
        :param num_features:    The total number of features
        :param model:           The model to be used for making predictions
        :return:                A `np.ndarray` or a `scipy.sparse`, shape `(num_examples, num_labels)`, that stores the
                                predictions for individual examples and labels
        """
        pass
