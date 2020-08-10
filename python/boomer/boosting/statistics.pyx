"""
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides wrappers for classes that allow to store gradients and Hessians that are computed according to a loss function
based on the current predictions of rules and the ground truth labels of the training examples.
"""


cdef class GradientStatistics(Statistics):
    """
    A wrapper for the C++ class `AbstractGradientStatistics`.
    """
    pass
