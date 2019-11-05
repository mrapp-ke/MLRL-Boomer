from boomer.algorithm._model cimport float64

cdef class Loss:

    cpdef float64[::1, :] calculate_initial_gradients(self, float64[::1, :] expected_scores)

    cpdef float64[::1, :] calculate_gradients(self, float64[::1, :] expected_scores, float64[::1, :] predicted_scores)

    cpdef float64[::1] calculate_optimal_scores(self, float64[::1, :] gradients)

    cpdef float64 evaluate_predictions(self, float64[::1] scores, float64[::1, :] gradients)


cdef class DecomposableLoss(Loss):

    cpdef float64[::1, :] calculate_initial_gradients(self, float64[::1, :] expected_scores)

    cpdef float64[::1, :] calculate_gradients(self, float64[::1, :] expected_scores, float64[::1, :] predicted_scores)

    cpdef float64[::1] calculate_optimal_scores(self, float64[::1, :] gradients)

    cpdef float64 evaluate_predictions(self, float64[::1] scores, float64[::1, :] gradients)


cdef class SquaredErrorLoss(DecomposableLoss):

    cpdef float64[::1, :] calculate_initial_gradients(self, float64[::1, :] expected_scores)

    cpdef float64[::1, :] calculate_gradients(self, float64[::1, :] expected_scores, float64[::1, :] predicted_scores)

    cpdef float64[::1] calculate_optimal_scores(self, float64[::1, :] gradients)

    cpdef float64 evaluate_predictions(self, float64[::1] scores, float64[::1, :] gradients)
