from libcpp.memory cimport shared_ptr


cdef extern from "cpp/label_wise_losses.h" namespace "boosting" nogil:

    cdef cppclass AbstractLabelWiseLoss:
        pass


    cdef cppclass LabelWiseLogisticLossImpl(AbstractLabelWiseLoss):
        pass


    cdef cppclass LabelWiseSquaredErrorLossImpl(AbstractLabelWiseLoss):
        pass


    cdef cppclass LabelWiseSquaredHingeLossImpl(AbstractLabelWiseLoss):

        # Functions:

        pair[float64, float64] calculateGradientAndHessian(AbstractLabelMatrix* labelMatrix, intp exampleIndex,
                                                           intp labelIndex, float64 predictedScore)


cdef class LabelWiseLoss:

    # Attributes:

    cdef shared_ptr[AbstractLabelWiseLoss] loss_function_ptr


cdef class LabelWiseLogisticLoss(LabelWiseLoss):
    pass


cdef class LabelWiseSquaredErrorLoss(LabelWiseLoss):
    pass


cdef class LabelWiseSquaredHingeLoss(LabelWiseLoss):
    pass
