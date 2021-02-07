from common.cython._types cimport uint8, uint32
from common.cython._data cimport CContiguousView
from common.cython.input cimport CContiguousFeatureMatrix, CContiguousFeatureMatrixImpl, CsrFeatureMatrix, \
    CsrFeatureMatrixImpl
from common.cython.model cimport RuleModel, RuleModelImpl

from libcpp.memory cimport unique_ptr


cdef extern from "common/output/predictor.hpp" nogil:

    cdef cppclass IPredictor[T]:

        # Functions:

        void predict(const CContiguousFeatureMatrixImpl& featureMatrix, CContiguousView[T]& predictionMatrix,
                     const RuleModelImpl& model)

        void predict(const CsrFeatureMatrixImpl& featureMatrix, CContiguousView[T]& predictionMatrix,
                     const RuleModelImpl& model)


cdef class Predictor:

    # Functions:

    cpdef object predict(self, CContiguousFeatureMatrix feature_matrix, RuleModel model)

    cpdef object predict_csr(self, CsrFeatureMatrix feature_matrix, RuleModel model)


cdef class AbstractClassificationPredictor(Predictor):

    # Attributes:

    cdef uint32 num_labels

    cdef unique_ptr[IPredictor[uint8]] predictor_ptr

    # Functions:

    cpdef object predict(self, CContiguousFeatureMatrix feature_matrix, RuleModel model)

    cpdef object predict_csr(self, CsrFeatureMatrix feature_matrix, RuleModel model)
