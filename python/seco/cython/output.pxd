from common.cython._types cimport uint8
from common.cython.output cimport AbstractClassificationPredictor, IPredictor


cdef extern from "seco/output/predictor_classification_label_wise.hpp" namespace "seco" nogil:

    cdef cppclass LabelWiseClassificationPredictorImpl"seco::LabelWiseClassificationPredictor"(IPredictor[uint8]):

        # Constructors:

        LabelWiseClassificationPredictorImpl(uint32 numThreads) except +


cdef class LabelWiseClassificationPredictor(AbstractClassificationPredictor):

    # Attributes:

    cdef uint32 num_threads
