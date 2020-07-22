"""
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides classes that allow to store the elements of confusion matrices that are computed independently for each label.
"""


cdef class LabelWiseStatistics(CoverageStatistics):
    """
    Allows to store the elements of confusion matrices that are computed independently for each label.
    """

    cdef void apply_default_prediction(self, LabelMatrix label_matrix, DefaultPrediction* default_prediction):
        pass

    cdef void reset_statistics(self):
        pass

    cdef void add_sampled_statistic(self, intp statistic_index, uint32 weight):
        pass

    cdef void update_covered_statistic(self, intp statistic_index, uint32 weight, bint remove):
        pass

    cdef RefinementSearch begin_search(self, intp[::1] label_indices):
        pass

    cdef void apply_prediction(self, intp statistic_index, intp[::1] label_indices, HeadCandidate* head):
        pass
