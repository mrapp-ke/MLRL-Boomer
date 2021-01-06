"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides classes for making predictions using rule-based models that have been learned by the boosting algorithm.
"""
from libcpp.memory cimport make_unique


cdef class ClassificationPredictor(Predictor):
    """
    A wrapper for the C++ class `ClassificationPredictor`.
    """

    def __cinit__(self, float64 threshold):
        self.predictor_ptr = make_unique[ClassificationPredictorImpl](threshold)

    cpdef object predict(self, float32[:, ::1] array, RuleModel model):
        # TODO Implement
        pass

    cpdef object predict_csr(self, float32[::1] data, uint32[::1] row_indices, uint32[::1] col_indices,
                             uint32 num_features, RuleModel model):
        # TODO Implement
        pass
