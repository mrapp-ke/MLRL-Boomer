"""
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides classes that allow to store gradients and Hessians that are calculated according to a (non-decomposable) loss
function that is applied example-wise.
"""


cdef class ExampleWiseStatistics(GradientStatistics):
    """
    Allows to store gradients and Hessians that are calculated according to a loss function that is applied
    example-wise.
    """

    def __cinit__(self, ExampleWiseLossFunction loss_function):
        """
        :param loss_function: The loss function to be used for calculating gradients and Hessians
        """
        self.loss_function = loss_function

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
