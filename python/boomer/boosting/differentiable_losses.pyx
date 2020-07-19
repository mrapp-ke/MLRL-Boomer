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

    cdef DefaultPrediction* calculate_default_prediction(self, LabelMatrix label_matrix):
        pass

    cdef void reset_examples(self):
        pass

    cdef void add_sampled_example(self, intp example_index, uint32 weight):
        # This function is equivalent to the function `update_covered_example`...
        self.update_covered_example(example_index, weight, False)

    cdef void update_covered_example(self, intp example_index, uint32 weight, bint remove):
        pass

    cdef RefinementSearch begin_search(self, intp[::1] label_indices):
        pass

    cdef void apply_prediction(self, intp example_index, intp[::1] label_indices, float64[::1] predicted_scores):
        pass
