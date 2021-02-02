from boomer.common._types cimport uint8, float64
from boomer.common._data cimport BinarySparseListVector
from boomer.common.output cimport AbstractClassificationPredictor, IPredictor

from libcpp.memory cimport unique_ptr


cdef extern from "cpp/output/predictor_classification_label_wise.h" namespace "boosting" nogil:

    cdef cppclass LabelWiseClassificationPredictorImpl"boosting::LabelWiseClassificationPredictor"(IPredictor[uint8]):

        # Constructors:

        LabelWiseClassificationPredictorImpl(float64 threshold)


cdef extern from "cpp/output/predictor_classification_example_wise.h" namespace "boosting" nogil:

    cdef cppclass ExampleWiseClassificationPredictorImpl"boosting::ExampleWiseClassificationPredictor"(
            IPredictor[uint8]):

        ctypedef BinarySparseListVector LabelVector

        # Functions:

        void addLabelVector(unique_ptr[LabelVector] labelVectorPtr)


cdef class LabelWiseClassificationPredictor(AbstractClassificationPredictor):

    # Attributes:

    cdef float64 threshold


cdef class ExampleWiseClassificationPredictor(AbstractClassificationPredictor):
    pass
