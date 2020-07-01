"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides base classes for all differentiable (surrogate) loss functions to be minimized locally by the rules learned by
a boosting algorithm.
"""


cdef class DifferentiableLoss(Loss):
    """
    A base class for all differentiable (surrogate) loss functions to be minimized locally by the rules learned by a
    boosting algorithm.
    """

    cdef DefaultPrediction calculate_default_prediction(self, uint8[::1, :] y):
        pass

    cdef void begin_instance_sub_sampling(self):
        pass

    cdef void update_sub_sample(self, intp example_index, uint32 weight, bint remove):
        pass

    cdef void begin_search(self, intp[::1] label_indices):
        pass

    cdef void update_search(self, intp example_index, uint32 weight):
        pass

    cdef void reset_search(self):
        pass

    cdef LabelWisePrediction calculate_label_wise_prediction(self, bint uncovered, bint accumulated):
        pass

    cdef Prediction calculate_example_wise_prediction(self, bint uncovered, bint accumulated):
        pass

    cdef void apply_prediction(self, intp example_index, intp[::1] label_indices, float64[::1] predicted_scores):
        pass

cdef class DecomposableDifferentiableLoss(DifferentiableLoss):
    """
    A base class for all (label-wise) decomposable differentiable loss functions.
    """

    cdef DefaultPrediction calculate_default_prediction(self, uint8[::1, :] y):
        pass

    cdef void begin_instance_sub_sampling(self):
        pass

    cdef void update_sub_sample(self, intp example_index, uint32 weight, bint remove):
        pass

    cdef void begin_search(self, intp[::1] label_indices):
        pass

    cdef void update_search(self, intp example_index, uint32 weight):
        pass

    cdef void reset_search(self):
        pass

    cdef LabelWisePrediction calculate_label_wise_prediction(self, bint uncovered, bint accumulated):
        pass

    cdef Prediction calculate_example_wise_prediction(self, bint uncovered, bint accumulated):
        # In case of a decomposable loss, the label-dependent predictions are the same as the label-independent
        # predictions...
        return self.calculate_label_wise_prediction(uncovered, accumulated)

    cdef void apply_prediction(self, intp example_index, intp[::1] label_indices, float64[::1] predicted_scores):
        pass


cdef class NonDecomposableDifferentiableLoss(DifferentiableLoss):
    """
    A base class for all (label-wise) non-decomposable differentiable loss functions.
    """

    cdef DefaultPrediction calculate_default_prediction(self, uint8[::1, :] y):
        pass

    cdef void begin_instance_sub_sampling(self):
        pass

    cdef void update_sub_sample(self, intp example_index, uint32 weight, bint remove):
        pass

    cdef void begin_search(self, intp[::1] label_indices):
        pass

    cdef void update_search(self, intp example_index, uint32 weight):
        pass

    cdef void reset_search(self):
        pass

    cdef LabelWisePrediction calculate_label_wise_prediction(self, bint uncovered, bint accumulated):
        pass

    cdef Prediction calculate_example_wise_prediction(self, bint uncovered, bint accumulated):
        pass

    cdef void apply_prediction(self, intp example_index, intp[::1] label_indices, float64[::1] predicted_scores):
        pass
