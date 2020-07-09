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

    cdef void reset_examples(self):
        pass

    cdef void update_covered_example(self, intp example_index, uint32 weight, bint remove):
        pass

    cdef void begin_search(self, intp[::1] label_indices):
        pass

    cdef void update_search(self, intp example_index, uint32 weight):
        pass

    cdef void reset_search(self):
        pass

    cdef LabelIndependentPrediction evaluate_label_independent_predictions(self, bint uncovered, bint accumulated):
        pass

    cdef Prediction evaluate_label_dependent_predictions(self, bint uncovered, bint accumulated):
        pass

    cdef void apply_prediction(self, intp example_index, intp[::1] label_indices, float64[::1] predicted_scores):
        pass


cdef class DecomposableCoverageLoss(CoverageLoss):
    """
    A base class for all (label-wise) decomposable coverage loss functions.
    """

    cdef float64[::1] calculate_default_scores(self, uint8[::1, :] y):
        pass

    cdef void reset_examples(self):
        pass

    cdef void update_covered_example(self, intp example_index, uint32 weight, bint remove):
        pass

    cdef void begin_search(self, intp[::1] label_indices):
        pass

    cdef void update_search(self, intp example_index, uint32 weight):
        pass

    cdef void reset_search(self):
        pass

    cdef LabelIndependentPrediction evaluate_label_independent_predictions(self, bint uncovered, bint accumulated):
        pass

    cdef Prediction evaluate_label_dependent_predictions(self, bint uncovered, bint accumulated):
        # In case of a decomposable loss, the label-dependent predictions are the same as the label-independent
        # predictions...
        return self.evaluate_label_independent_predictions(uncovered, accumulated)

    cdef void apply_prediction(self, intp example_index, intp[::1] label_indices, float64[::1] predicted_scores):
        pass
