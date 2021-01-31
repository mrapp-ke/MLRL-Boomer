"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)
"""
from boomer.common._types cimport uint32

from libcpp.memory cimport unique_ptr, make_unique


cdef class ClassificationPredictor(AbstractClassificationPredictor):
    """
    A wrapper for the C++ class `ClassificationPredictor`.
    """

    def __cinit__(self, uint32 num_labels, float64 threshold):
        """
        :param num_labels:  The total number of available labels
        :param thresholds:  The threshold to be used for making predictions
        """
        self.num_labels = num_labels
        self.threshold = threshold
        self.predictor_ptr = <unique_ptr[IPredictor[uint8]]>make_unique[ClassificationPredictorImpl](threshold)

    def __reduce__(self):
        return (ClassificationPredictor, (self.num_labels, self.threshold))
