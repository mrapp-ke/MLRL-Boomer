from boomer.common._arrays cimport intp, float64
from boomer.common.statistics cimport LabelMatrix, AbstractLabelMatrix


cdef extern from "cpp/example_wise_losses.h" namespace "losses":

    cdef cppclass AbstractExampleWiseLoss:

        # Functions:

        void calculateGradientsAndHessians(AbstractLabelMatrix* labelMatrix, intp exampleIndex,
                                           float64* predictedScores, float64* gradients, float64* hessians) nogil


    cdef cppclass ExampleWiseLogisticLossImpl(AbstractExampleWiseLoss):

        # Functions:

        void calculateGradientsAndHessians(AbstractLabelMatrix* labelMatrix, intp exampleIndex,
                                           float64* predictedScores, float64* gradients, float64* hessians) nogil


cdef class ExampleWiseLoss:

    # Functions:

    cdef void calculate_gradients_and_hessians(self, LabelMatrix label_matrix, intp example_index,
                                               float64* predicted_scores, float64[::1] gradients, float64[::1] hessians)


cdef class ExampleWiseLogisticLoss(ExampleWiseLoss):

    # Functions:

    cdef void calculate_gradients_and_hessians(self, LabelMatrix label_matrix, intp example_index,
                                               float64* predicted_scores, float64[::1] gradients, float64[::1] hessians)
