from boomer.algorithm._model cimport float64

cdef class Loss:

    cdef float64[::1, :] calculate_initial_gradients(self, float64[::1, :] expected_scores)

    cdef float64[::1, :] calculate_gradients(self, float64[::1, :] expected_scores, float64[::1, :] predicted_scores)

    cdef float64[::1] calculate_optimal_scores(self, float64[::1, :] gradients)

    cdef float64 evaluate_predictions(self, float64[::1] scores, float64[::1, :] gradients)


cdef class DecomposableLoss(Loss):

    cdef float64[::1, :] calculate_initial_gradients(self, float64[::1, :] expected_scores)

    cdef float64[::1, :] calculate_gradients(self, float64[::1, :] expected_scores, float64[::1, :] predicted_scores)

    cdef float64[::1] calculate_optimal_scores(self, float64[::1, :] gradients)

    cdef float64 evaluate_predictions(self, float64[::1] scores, float64[::1, :] gradients)


cdef class SquaredErrorLoss(DecomposableLoss):

    cdef float64[::1, :] calculate_initial_gradients(self, float64[::1, :] expected_scores)

    cdef float64[::1, :] calculate_gradients(self, float64[::1, :] expected_scores, float64[::1, :] predicted_scores)

    cdef float64[::1] calculate_optimal_scores(self, float64[::1, :] gradients)

    cdef float64 evaluate_predictions(self, float64[::1] scores, float64[::1, :] gradients)
