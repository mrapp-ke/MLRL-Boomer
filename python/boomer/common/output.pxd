from boomer.common._types cimport uint32, float32
from boomer.common.model cimport RuleModel


cdef class Predictor:

    # Functions:

    cpdef object predict(self, float32[:, ::1] array, RuleModel model)

    cpdef object predict_csr(self, float32[::1] data, uint32[::1] row_indices, uint32[::1] col_indices,
                             uint32 num_features, RuleModel model)
