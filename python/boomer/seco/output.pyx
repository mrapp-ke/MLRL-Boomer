"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides classes for making predictions using rule-based models that have been learned by the seperate-and-conquer
algorithm.
"""
from boomer.common._types cimport uint32

from libcpp.memory cimport unique_ptr, make_unique


cdef class ClassificationPredictor(AbstractClassificationPredictor):
    """
    A wrapper for the C++ class `ClassificationPredictor`.
    """

    def __cinit__(self, uint32 num_labels):
        """
        :param num_labels: The total number of available labels
        """
        self.num_labels = num_labels
        self.predictor_ptr = <unique_ptr[IPredictor[uint8]]>make_unique[ClassificationPredictorImpl]()
