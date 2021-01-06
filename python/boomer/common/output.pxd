from boomer.common._types cimport uint32, float32
from boomer.common._data cimport CContiguousView
from boomer.common.input cimport CContiguousFeatureMatrixImpl, CsrFeatureMatrixImpl
from boomer.common.model cimport RuleModel, RuleModelImpl


cdef extern from "cpp/output/predictor.h" nogil:

    cdef cppclass IPredictor[T]:

        # Functions:

        void predict(const CContiguousFeatureMatrixImpl& featureMatrix, CContiguousView[T]& predictionMatrix,
                     const RuleModelImpl& model)

        void predict(const CsrFeatureMatrixImpl& featureMatrix, CContiguousView[T]& predictionMatrix,
                     const RuleModelImpl& model)


cdef class Predictor:

    # Functions:

    cpdef object predict(self, float32[:, ::1] array, RuleModel model)

    cpdef object predict_csr(self, float32[::1] data, uint32[::1] row_indices, uint32[::1] col_indices,
                             uint32 num_features, RuleModel model)
