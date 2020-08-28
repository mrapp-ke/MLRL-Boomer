"""
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides wrappers for classes that allow to store statistics about the labels of training examples.
"""


cdef class StatisticsFactory:
    """
    A wrapper for the C++ class `AbstractStatisticsFactory`.
    """

    cdef AbstractStatistics* create(self, RandomAccessLabelMatrix label_matrix):
        """
        Creates a new instance of the class `AbstractStatistics`.

        :param:     A `RandomAccessLabelMatrix` that provides access to the labels of the training examples
        :return:    A pointer to an object of type `AbstractStatistics` that has been created
        """
        pass
