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

    cdef DefaultPrediction calculate_default_prediction(self, LabelMatrix label_matrix):
        pass

    cdef void begin_instance_sub_sampling(self):
        pass

    cdef void update_sub_sample(self, intp example_index, uint32 weight, bint remove):
        pass

    cdef RefinementSearch begin_search(self, intp[::1] label_indices):
        pass

    cdef void apply_prediction(self, intp example_index, intp[::1] label_indices, float64[::1] predicted_scores):
        pass
