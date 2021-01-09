from boomer.common._types cimport uint8
from boomer.common.input cimport CContiguousFeatureMatrix, CsrFeatureMatrix
from boomer.common.model cimport RuleModel
from boomer.common.output cimport AbstractClassificationPredictor, IPredictor


cdef extern from "cpp/output/predictor_classification.h" namespace "seco" nogil:

    cdef cppclass ClassificationPredictorImpl"seco::ClassificationPredictor"(IPredictor[uint8]):
        pass


cdef class ClassificationPredictor(AbstractClassificationPredictor):

    cpdef object predict(self, CContiguousFeatureMatrix feature_matrix, RuleModel model)

    cpdef object predict_csr(self, CsrFeatureMatrix feature_matrix, RuleModel model)
