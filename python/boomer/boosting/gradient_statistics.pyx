"""
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides base classes for all classes that allow to store gradients and Hessians that are computed according to a loss
function based on the current predictions of rules and the ground truth labels of the training examples.
"""


cdef class GradientStatistics(Statistics):
    """
    A base class for all classes that store gradients and Hessians.
    """

    cdef void reset_statistics(self):
        pass

    cdef void add_sampled_statistic(self, intp statistic_index, uint32 weight):
        # This function is equivalent to the function `update_covered_statistic`...
        self.update_covered_statistic(statistic_index, weight, False)

    cdef void update_covered_statistic(self, intp statistic_index, uint32 weight, bint remove):
        pass

    cdef RefinementSearch begin_search(self, intp[::1] label_indices):
        pass

    cdef void apply_prediction(self, intp statistic_index, intp[::1] label_indices, HeadCandidate* head):
        pass
