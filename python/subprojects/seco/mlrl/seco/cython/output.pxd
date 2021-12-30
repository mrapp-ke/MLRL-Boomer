from mlrl.common.cython._types cimport uint8, uint32
from mlrl.common.cython.output cimport BinaryPredictor, ClassificationPredictorFactory, IClassificationPredictorFactory


cdef extern from "seco/output/predictor_classification_label_wise.hpp" namespace "seco" nogil:

    cdef cppclass LabelWiseClassificationPredictorFactoryImpl"seco::LabelWiseClassificationPredictorFactory"(
            IClassificationPredictorFactory):

        # Constructors:

        LabelWiseClassificationPredictorFactoryImpl(uint32 numThreads) except +


cdef class LabelWiseClassificationPredictorFactory(ClassificationPredictorFactory):

    # Attributes:

    cdef uint32 num_threads
