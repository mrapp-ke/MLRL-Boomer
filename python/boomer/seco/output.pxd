from boomer.common._types cimport uint8
from boomer.common.output cimport AbstractClassificationPredictor, IPredictor


cdef extern from "cpp/output/predictor_classification_label_wise.h" namespace "seco" nogil:

    cdef cppclass LabelWiseClassificationPredictorImpl"seco::LabelWiseClassificationPredictor"(IPredictor[uint8]):
        pass


cdef class LabelWiseClassificationPredictor(AbstractClassificationPredictor):
    pass
