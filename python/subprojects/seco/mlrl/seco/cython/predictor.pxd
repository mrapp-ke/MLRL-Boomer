from mlrl.common.cython._types cimport uint32


cdef extern from "seco/output/predictor_classification_label_wise.hpp" namespace "seco" nogil:

    cdef cppclass ILabelWiseClassificationPredictorConfig:

        # Functions:

        uint32 getNumThreads() const

        ILabelWiseClassificationPredictorConfig& setNumThreads(uint32 numThreads) except +


cdef class LabelWiseClassificationPredictorConfig:

    # Attributes:

    cdef ILabelWiseClassificationPredictorConfig* config_ptr
