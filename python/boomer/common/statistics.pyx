"""
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides wrappers for classes that allow to store statistics about the labels of training examples.
"""


cdef class StatisticsFactory:
    """
    A factory that allow to create instances of the class `AbstractStatistics`.
    """

    cdef AbstractStatistics* create_initial_statistics(self, LabelMatrix label_matrix):
        """
        Creates a new instance of the class `AbstractStatistics`, representing the initial statistics as computed based
        on the ground truth labels.

        :param label_matrix:    A `LabelMatrix` that provides access to the labels of the training examples
        :return:                A pointer to an object of type `AbstractStatistics` that has been created
        """
        pass
