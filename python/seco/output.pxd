from common._types cimport uint8
from common.output cimport AbstractClassificationPredictor, IPredictor


cdef extern from "seco/output/predictor_classification_label_wise.hpp" namespace "seco" nogil:

    cdef cppclass LabelWiseClassificationPredictorImpl"seco::LabelWiseClassificationPredictor"(IPredictor[uint8]):
        pass


cdef class LabelWiseClassificationPredictor(AbstractClassificationPredictor):
    pass
