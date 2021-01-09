"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides classes for making predictions using rule-based models that have been learned by the seperate-and-conquer
algorithm.
"""
from boomer.common._arrays cimport c_matrix_uint8
from boomer.common._data cimport CContiguousView
from boomer.common.input cimport CContiguousFeatureMatrixImpl

from libcpp.memory cimport make_unique

from cython.operator cimport dereference

import numpy as np


cdef class ClassificationPredictor(Predictor):
    """
    A wrapper for the C++ class `ClassificationPredictor`.
    """

    def __cinit__(self, uint32 num_labels):
        """
        :param num_labels: The total number of available labels
        """
        self.num_labels = num_labels
        self.predictor_ptr = make_unique[ClassificationPredictorImpl]()

    cpdef object predict(self, float32[:, ::1] array, RuleModel model):
        cdef uint32 num_examples = array.shape[0]
        cdef uint32 num_features = array.shape[1]
        cdef uint32 num_labels = self.num_labels
        cdef uint8[:, ::1] prediction_matrix = c_matrix_uint8(num_examples, num_labels)
        cdef unique_ptr[CContiguousView[uint8]] view_ptr = make_unique[CContiguousView[uint8]](
            num_examples, num_labels, &prediction_matrix[0, 0])
        cdef unique_ptr[CContiguousFeatureMatrixImpl] feature_matrix_ptr = make_unique[CContiguousFeatureMatrixImpl](
            num_examples, num_features, &array[0, 0])
        self.predictor_ptr.get().predict(dereference(feature_matrix_ptr.get()), dereference(view_ptr.get()),
                                         dereference(model.model_ptr.get()))
        return np.asarray(prediction_matrix)

    cpdef object predict_csr(self, float32[::1] data, uint32[::1] row_indices, uint32[::1] col_indices,
                             uint32 num_features, RuleModel model):
        # TODO Implement
        pass