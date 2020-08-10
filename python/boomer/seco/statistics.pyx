"""
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides wrappers for classes that allow to store the elements of confusion matrices that are computed based on a weight
matrix and the ground truth labels of the training examples.
"""


cdef class CoverageStatistics(Statistics):
    """
    A wrapper for the C++ class `AbstractCoverageStatistics`.
    """
    pass
