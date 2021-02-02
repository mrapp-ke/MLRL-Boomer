"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)
"""
from boomer.common._types cimport uint32

from libcpp.memory cimport unique_ptr, make_unique


cdef class LabelWiseClassificationPredictor(AbstractClassificationPredictor):
    """
    A wrapper for the C++ class `LabelWiseClassificationPredictor`.
    """

    def __cinit__(self, uint32 num_labels):
        """
        :param num_labels: The total number of available labels
        """
        self.num_labels = num_labels
        self.predictor_ptr = <unique_ptr[IPredictor[uint8]]>make_unique[LabelWiseClassificationPredictorImpl]()

    def __reduce__(self):
        return (LabelWiseClassificationPredictor, (self.num_labels,))
