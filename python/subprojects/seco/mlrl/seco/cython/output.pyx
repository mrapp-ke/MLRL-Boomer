"""
@author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from libcpp.memory cimport unique_ptr, make_unique


cdef class LabelWiseClassificationPredictorFactory(ClassificationPredictorFactory):
    """
    A wrapper for the C++ class `LabelWiseClassificationPredictorFactory`.
    """

    def __cinit__(self, uint32 num_labels, uint32 num_threads):
        """
        :param num_labels:  The total number of available labels
        :param num_threads: The number of CPU threads to be used to make predictions for different query examples in
                            parallel. Must be at least 1
        """
        self.num_labels = num_labels
        self.num_threads = num_threads
        self.predictor_factory_ptr = <unique_ptr[IClassificationPredictorFactory]>make_unique[LabelWiseClassificationPredictorFactoryImpl](
            num_threads)

    def __reduce__(self):
        return (LabelWiseClassificationPredictorFactory, (self.num_labels, self.num_threads))
