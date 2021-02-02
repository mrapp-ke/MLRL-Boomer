from boomer.common._types cimport uint8, float64
from boomer.common.input cimport CContiguousFeatureMatrix, CsrFeatureMatrix
from boomer.common.model cimport RuleModel
from boomer.common.output cimport AbstractClassificationPredictor, IPredictor


cdef extern from "cpp/output/predictor_classification_label_wise.h" namespace "boosting" nogil:

    cdef cppclass LabelWiseClassificationPredictorImpl"boosting::LabelWiseClassificationPredictor"(IPredictor[uint8]):

        LabelWiseClassificationPredictorImpl(float64 threshold)


cdef class LabelWiseClassificationPredictor(AbstractClassificationPredictor):

    # Attributes:

    cdef float64 threshold

    # Functions:

    cpdef object predict(self, CContiguousFeatureMatrix feature_matrix, RuleModel model)

    cpdef object predict_csr(self, CsrFeatureMatrix feature_matrix, RuleModel model)
