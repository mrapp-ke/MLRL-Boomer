from mlrl.common.cython._types cimport uint32


cdef extern from "seco/output/predictor_classification_label_wise.hpp" namespace "seco" nogil:

    cdef cppclass LabelWiseClassificationPredictorConfigImpl"seco::LabelWiseClassificationPredictorConfig":

        # Functions:

        uint32 getNumThreads() const

        LabelWiseClassificationPredictorConfigImpl& setNumThreads(uint32 numThreads) except +


cdef class LabelWiseClassificationPredictorConfig:

    # Attributes:

    cdef LabelWiseClassificationPredictorConfigImpl* config_ptr
