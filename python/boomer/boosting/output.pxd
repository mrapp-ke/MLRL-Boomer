from boomer.common._types cimport uint8, float64
from boomer.common.input cimport CContiguousFeatureMatrix, CsrFeatureMatrix
from boomer.common.model cimport RuleModel
from boomer.common.output cimport AbstractClassificationPredictor, IPredictor


cdef extern from "cpp/output/predictor_classification.h" namespace "boosting" nogil:

    cdef cppclass ClassificationPredictorImpl"boosting::ClassificationPredictor"(IPredictor[uint8]):

        ClassificationPredictorImpl(float64 threshold)


cdef class ClassificationPredictor(AbstractClassificationPredictor):

    # Functions:

    cpdef object predict(self, CContiguousFeatureMatrix feature_matrix, RuleModel model)

    cpdef object predict_csr(self, CsrFeatureMatrix feature_matrix, RuleModel model)
