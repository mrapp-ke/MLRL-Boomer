from boomer.common._types cimport uint8, float64
from boomer.common._measures cimport IMeasure
from boomer.common.input cimport LabelVector
from boomer.common.output cimport AbstractClassificationPredictor, IPredictor

from libcpp.memory cimport unique_ptr, shared_ptr


cdef extern from "cpp/output/predictor_classification_label_wise.h" namespace "boosting" nogil:

    cdef cppclass LabelWiseClassificationPredictorImpl"boosting::LabelWiseClassificationPredictor"(IPredictor[uint8]):

        # Constructors:

        LabelWiseClassificationPredictorImpl(float64 threshold) except +


ctypedef void (*LabelVectorVisitor)(const LabelVector&)


cdef extern from "cpp/output/predictor_classification_example_wise.h" namespace "boosting" nogil:

    cdef cppclass ExampleWiseClassificationPredictorImpl"boosting::ExampleWiseClassificationPredictor"(
            IPredictor[uint8]):

        # Constructors:

        ExampleWiseClassificationPredictorImpl(shared_ptr[IMeasure] measurePtr) except +

        # Functions:

        void addLabelVector(unique_ptr[LabelVector] labelVectorPtr)

        void visit(LabelVectorVisitor)


cdef class LabelWiseClassificationPredictor(AbstractClassificationPredictor):

    # Attributes:

    cdef float64 threshold


cdef class ExampleWiseClassificationPredictor(AbstractClassificationPredictor):
    pass
