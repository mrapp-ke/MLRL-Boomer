from libcpp.memory cimport shared_ptr


cdef extern from "cpp/label_wise_losses.h" namespace "boosting" nogil:

    cdef cppclass ILabelWiseLoss:
        pass


    cdef cppclass LabelWiseLogisticLossImpl(ILabelWiseLoss):
        pass


    cdef cppclass LabelWiseSquaredErrorLossImpl(ILabelWiseLoss):
        pass


cdef class LabelWiseLoss:

    # Attributes:

    cdef shared_ptr[ILabelWiseLoss] loss_function_ptr


cdef class LabelWiseLogisticLoss(LabelWiseLoss):
    pass

cdef class LabelWiseSquaredErrorLoss(LabelWiseLoss):
    pass
