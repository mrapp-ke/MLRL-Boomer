"""
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides base classes for all classes that allow to store gradients and Hessians that are computed according to a loss
function based on the current predictions of rules and the ground truth labels of the training examples.
"""


cdef class GradientStatistics(Statistics):
    """
    A base class for all classes that store gradients and Hessians.
    """

    cdef void apply_default_prediction(self, LabelMatrix label_matrix, DefaultPrediction* default_prediction):
        pass

    cdef void reset_sampled_statistics(self):
        # This function is equivalent to the function `reset_covered_statistics`...
        self.reset_covered_statistics()

    cdef void add_sampled_statistic(self, intp statistic_index, uint32 weight):
        # This function is equivalent to the function `update_covered_statistic`...
        self.update_covered_statistic(statistic_index, weight, False)

    cdef void reset_covered_statistics(self):
        pass

    cdef void update_covered_statistic(self, intp statistic_index, uint32 weight, bint remove):
        pass

    cdef AbstractRefinementSearch* begin_search(self, intp[::1] label_indices):
        pass

    cdef void apply_prediction(self, intp statistic_index, intp[::1] label_indices, HeadCandidate* head):
        pass
