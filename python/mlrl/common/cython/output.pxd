from mlrl.common.cython._types cimport uint8, uint32, float64
from mlrl.common.cython._data cimport CContiguousView
from mlrl.common.cython.input cimport CContiguousFeatureMatrixImpl, CsrFeatureMatrixImpl, LabelVectorSetImpl
from mlrl.common.cython.model cimport RuleModelImpl

from libcpp.memory cimport unique_ptr


cdef extern from "common/output/prediction_matrix_sparse.hpp" nogil:

    cdef cppclass SparsePredictionMatrix[T]:
        pass


cdef extern from "common/output/predictor_dense.hpp" nogil:

    cdef cppclass IDensePredictor[T]:

        # Functions:

        void predict(const CContiguousFeatureMatrixImpl& featureMatrix, CContiguousView[T]& predictionMatrix,
                     const RuleModelImpl& model, const LabelVectorSetImpl* labelVectors)

        void predict(const CsrFeatureMatrixImpl& featureMatrix, CContiguousView[T]& predictionMatrix,
                     const RuleModelImpl& model, const LabelVectorSetImpl* labelVectors)


cdef extern from "common/output/predictor_sparse.hpp" nogil:

    cdef cppclass ISparsePredictor[T]:

        # Functions:

        unique_ptr[SparsePredictionMatrix[T]] predict(const CContiguousFeatureMatrixImpl& featureMatrix,
                                                      uint32 numLabels, const RuleModelImpl& model,
                                                      const LabelVectorSetImpl* labelVectors)

        unique_ptr[SparsePredictionMatrix[T]] predict(const CsrFeatureMatrixImpl& featureMatrix, uint32 numLabels,
                                                      const RuleModelImpl& model,
                                                      const LabelVectorSetImpl* labelVectors)


cdef class DensePredictor:
    pass


cdef class AbstractNumericalPredictor(DensePredictor):

    # Attributes:

    cdef uint32 num_labels

    cdef unique_ptr[IDensePredictor[float64]] predictor_ptr


cdef class AbstractBinaryPredictor(DensePredictor):

    # Attributes:

    cdef uint32 num_labels

    cdef unique_ptr[IDensePredictor[uint8]] predictor_ptr
