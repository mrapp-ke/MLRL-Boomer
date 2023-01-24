"""
@author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from mlrl.common.cython._arrays cimport array_uint32, c_matrix_uint8, c_matrix_float64

from scipy.sparse import csr_matrix
import numpy as np


cdef class BinaryPredictor:
    """
    Allows to predict labels for given query examples.
    """

    def predict(self) -> np.ndarray:
        """
        Obtains and returns predictions for all query examples.

        :return: A `numpy.ndarray` of type `uint8`, shape `(num_examples, num_labels)`, that stores the predictions
        """
        cdef unique_ptr[DensePredictionMatrix[uint8]] prediction_matrix_ptr = self.predictor_ptr.get().predict()
        cdef uint32 num_rows = prediction_matrix_ptr.get().getNumRows()
        cdef uint32 num_cols = prediction_matrix_ptr.get().getNumCols()
        cdef uint8* array = prediction_matrix_ptr.get().release()
        cdef uint8[:, ::1] prediction_matrix = c_matrix_uint8(array, num_rows, num_cols)
        return np.asarray(prediction_matrix)


cdef class SparseBinaryPredictor:
    """
    Allows to predict sparse labels for given query examples.
    """

    def predict(self) -> csr_matrix:
        """
        Obtains and returns predictions for all query examples.

        :return: A `scipy.sparse.csr_matrix` of type `uint8`, shape `(num_examples, num_labels)` that stores the
                 predictions
        """
        cdef unique_ptr[BinarySparsePredictionMatrix] prediction_matrix_ptr = self.predictor_ptr.get().predict()
        cdef uint32 num_rows = prediction_matrix_ptr.get().getNumRows()
        cdef uint32 num_cols = prediction_matrix_ptr.get().getNumCols()
        cdef uint32 num_non_zero_elements = prediction_matrix_ptr.get().getNumNonZeroElements()
        cdef uint32* row_indices = prediction_matrix_ptr.get().releaseRowIndices()
        cdef uint32* col_indices = prediction_matrix_ptr.get().releaseColIndices()
        data = np.ones(shape=(num_non_zero_elements), dtype=np.uint8) if num_non_zero_elements > 0 else np.asarray([])
        indices = np.asarray(array_uint32(col_indices, num_non_zero_elements) if num_non_zero_elements > 0 else [])
        indptr = np.asarray(array_uint32(row_indices, num_rows + 1))
        return csr_matrix((data, indices, indptr), shape=(num_rows, num_cols))


cdef class ScorePredictor:
    """
    Allows to predict regression scores for given query examples.
    """

    def predict(self) -> np.ndarray:
        """
        Obtains and returns predictions for all query examples.

        :return: A `numpy.ndarray` of type `float64`, shape `(num_examples, num_labels)`, that stores the predictions
        """
        cdef unique_ptr[DensePredictionMatrix[float64]] prediction_matrix_ptr = self.predictor_ptr.get().predict()
        cdef uint32 num_rows = prediction_matrix_ptr.get().getNumRows()
        cdef uint32 num_cols = prediction_matrix_ptr.get().getNumCols()
        cdef float64* array = prediction_matrix_ptr.get().release()
        cdef float64[:, ::1] prediction_matrix = c_matrix_float64(array, num_rows, num_cols)
        return np.asarray(prediction_matrix)


cdef class ProbabilityPredictor:
    """
    Allows to predict probability estimates for given query examples.
    """

    def predict(self) -> np.ndarray:
        """
        Obtains and returns predictions for all query examples.

        :return: A `numpy.ndarray` of type `float64`, shape `(num_examples, num_labels)`, that stores the predictions
        """
        cdef unique_ptr[DensePredictionMatrix[float64]] prediction_matrix_ptr = self.predictor_ptr.get().predict()
        cdef uint32 num_rows = prediction_matrix_ptr.get().getNumRows()
        cdef uint32 num_cols = prediction_matrix_ptr.get().getNumCols()
        cdef float64* array = prediction_matrix_ptr.get().release()
        cdef float64[:, ::1] prediction_matrix = c_matrix_float64(array, num_rows, num_cols)
        return np.asarray(prediction_matrix)
