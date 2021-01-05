from boomer.common._types cimport uint32, float32, float64
from boomer.common.rules cimport RuleModel


cdef class Predictor:

    # Functions:

    cpdef object predict(self, float32[:, ::1] x, RuleModel model)

    cpdef object predict_csr(self, float32[::1] x_data, uint32[::1] x_row_indices, uint32[::1] x_col_indices,
                             uint32 num_features, RuleModel model)


cdef class DensePredictor(Predictor):

    # Attributes:

    cdef uint32 num_labels

    cdef TransformationFunction transformation_function

    # Functions:

    cpdef object predict(self, float32[:, ::1] x, RuleModel model)

    cpdef object predict_csr(self, float32[::1] x_data, uint32[::1] x_row_indices, uint32[::1] x_col_indices,
                             uint32 num_features, RuleModel model)

cdef class TransformationFunction:

    # Functions:

    cdef object transform_matrix(self, float64[:, ::1] m)


cdef class ThresholdFunction(TransformationFunction):

    # Attributes

    cdef readonly float64 threshold

    # Functions:

    cdef object transform_matrix(self, float64[:, ::1] m)
