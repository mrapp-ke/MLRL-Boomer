from mlrl.common.cython._types cimport uint8, uint32, float64
from mlrl.common.cython._data cimport CContiguousView
from mlrl.common.cython.input cimport CContiguousFeatureMatrixImpl, CsrFeatureMatrixImpl, LabelVectorSetImpl
from mlrl.common.cython.model cimport RuleModelImpl

from libcpp.memory cimport unique_ptr


cdef extern from "common/output/predictor.hpp" nogil:

    cdef cppclass IPredictor[T]:

        # Functions:

        void predict(const CContiguousFeatureMatrixImpl& featureMatrix, CContiguousView[T]& predictionMatrix,
                     const RuleModelImpl& model, const LabelVectorSetImpl* labelVectors)

        void predict(const CsrFeatureMatrixImpl& featureMatrix, CContiguousView[T]& predictionMatrix,
                     const RuleModelImpl& model, const LabelVectorSetImpl* labelVectors)


cdef class Predictor:
    pass


cdef class AbstractNumericalPredictor(Predictor):

    # Attributes:

    cdef uint32 num_labels

    cdef unique_ptr[IPredictor[float64]] predictor_ptr


cdef class AbstractBinaryPredictor(Predictor):

    # Attributes:

    cdef uint32 num_labels

    cdef unique_ptr[IPredictor[uint8]] predictor_ptr
