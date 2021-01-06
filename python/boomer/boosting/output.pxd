from boomer.common._types cimport uint32, float32, float64
from boomer.common.model cimport RuleModel
from boomer.common.output cimport Predictor, IPredictor

from libcpp.memory cimport unique_ptr


cdef extern from "cpp/output/predictor_classification.h" namespace "boosting" nogil:

    cdef cppclass ClassificationPredictorImpl"boosting::ClassificationPredictor"(IPredictor):

        ClassificationPredictorImpl(float64 threshold)


cdef class ClassificationPredictor(Predictor):

    # Attributes:

    cdef unique_ptr[ClassificationPredictorImpl] predictor_ptr

    # Functions:

    cpdef object predict(self, float32[:, ::1] array, RuleModel model)

    cpdef object predict_csr(self, float32[::1] data, uint32[::1] row_indices, uint32[::1] col_indices,
                             uint32 num_features, RuleModel model)
