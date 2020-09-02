"""
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides wrappers for classes that allow to store statistics about the labels of training examples.
"""


cdef class StatisticsProvider:
    """
    Provides access to an instance of the class `AbstractStatistics`.
    """

    cdef AbstractStatistics* get(self, LabelMatrix label_matrix):
        """
        Returns an instance of the class `AbstractStatistics`. If such an instance has not been created yet, it will be
        initialized based on the given label matrix. Otherwise, the instance which has been created earlier will be
        returned.

        :param label_matrix:    A `LabelMatrix` that provides access to the labels of the training examples
        :return:                A pointer to an object of type `AbstractStatistics`
        """
        pass
