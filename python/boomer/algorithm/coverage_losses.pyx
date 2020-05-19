"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides base classes for all (surrogate) loss functions to be minimized locally by the rules learned by an algorithm
based on sequential covering, such as e.g. a separate-and-conquer algorithm.
"""


cdef class CoverageLoss(Loss):
    """
    A base class for all (surrogate) loss functions to be minimized locally by the rules learned by an algorithm based
    on sequential covering, such as e.g. a separate-and-conquer algorithm.
    """

    cdef float64[::1] calculate_default_scores(self, uint8[::1, :] y):
        pass

    cdef void set_sub_sample(self, intp[::1] example_indices, uint32[::1] weights):
        pass

    cdef void remove_from_sub_sample(self, intp[::1] example_indices, uint32[::1] weights):
        pass

    cdef void begin_search(self, intp[::1] label_indices):
        pass

    cdef void update_search(self, intp example_index, uint32 weight):
        pass

    cdef LabelIndependentPrediction evaluate_label_independent_predictions(self, bint uncovered):
        pass

    cdef Prediction evaluate_label_dependent_predictions(self, bint uncovered):
        pass

    cdef void apply_predictions(self, intp[::1] covered_example_indices, intp[::1] label_indices,
                                float64[::1] predicted_scores):
        pass

cdef class DecomposableCoverageLoss(CoverageLoss):
    """
    A base class for all (label-wise) decomposable coverage loss functions.
    """

    cdef float64[::1] calculate_default_scores(self, uint8[::1, :] y):
        pass

    cdef void set_sub_sample(self, intp[::1] example_indices, uint32[::1] weights):
        pass

    cdef void remove_from_sub_sample(self, intp[::1] example_indices, uint32[::1] weights):
        pass

    cdef void begin_search(self, intp[::1] label_indices):
        pass

    cdef void update_search(self, intp example_index, uint32 weight):
        pass

    cdef LabelIndependentPrediction evaluate_label_independent_predictions(self, bint uncovered):
        pass

    cdef Prediction evaluate_label_dependent_predictions(self, bint uncovered):
        # In case of a decomposable loss, the label-dependent predictions are the same as the label-independent
        # predictions...
        return self.evaluate_label_independent_predictions(uncovered)

    cdef void apply_predictions(self, intp[::1] covered_example_indices, intp[::1] label_indices,
                                float64[::1] predicted_scores):
        pass
