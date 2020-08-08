"""
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides base classes for all classes that allow to store the elements of confusion matrices that are computed based on
a weight matrix and the ground truth labels of the training examples.
"""


cdef class CoverageStatistics(Statistics):
    """
    A base class for all classes that allow to store the elements of confusion matrices.
    """

    cdef float64 get_sum_uncovered_labels(self):
        """
        Returns the sum of weights of all labels that remain to be covered.

        :return: The sum of weights of all labels that remain to be covered
        """
        pass

    cdef void apply_default_prediction(self, LabelMatrix label_matrix, DefaultPrediction* default_prediction):
        pass

    cdef void reset_sampled_statistics(self):
        pass

    cdef void add_sampled_statistic(self, intp statistic_index, uint32 weight):
        pass

    cdef void reset_covered_statistics(self):
        pass

    cdef void update_covered_statistic(self, intp statistic_index, uint32 weight, bint remove):
        pass

    cdef AbstractRefinementSearch* begin_search(self, intp[::1] label_indices):
        pass

    cdef void apply_prediction(self, intp statistic_index, intp[::1] label_indices, HeadCandidate* head):
        pass
