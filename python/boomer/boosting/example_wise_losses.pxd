from boomer.common._arrays cimport intp, float64
from boomer.common.input_data cimport AbstractLabelMatrix

from libcpp.memory cimport shared_ptr


cdef extern from "cpp/example_wise_losses.h" namespace "boosting" nogil:

    cdef cppclass AbstractExampleWiseLoss:

        # Functions:

        void calculateGradientsAndHessians(AbstractLabelMatrix* labelMatrix, intp exampleIndex,
                                           float64* predictedScores, float64* gradients, float64* hessians)


    cdef cppclass ExampleWiseLogisticLossImpl(AbstractExampleWiseLoss):

        # Functions:

        void calculateGradientsAndHessians(AbstractLabelMatrix* labelMatrix, intp exampleIndex,
                                           float64* predictedScores, float64* gradients, float64* hessians)


cdef class ExampleWiseLoss:

    # Attributes:

    cdef shared_ptr[AbstractExampleWiseLoss] loss_function_ptr


cdef class ExampleWiseLogisticLoss(ExampleWiseLoss):
    pass
