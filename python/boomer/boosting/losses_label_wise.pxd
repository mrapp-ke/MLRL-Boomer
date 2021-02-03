from boomer.common._measures cimport IMeasure

from libcpp.memory cimport shared_ptr


cdef extern from "cpp/losses/loss_label_wise.h" namespace "boosting" nogil:

    cdef cppclass ILabelWiseLoss(IMeasure):
        pass


cdef extern from "cpp/losses/loss_label_wise_logistic.h" namespace "boosting" nogil:

    cdef cppclass LabelWiseLogisticLossImpl"boosting::LabelWiseLogisticLoss"(ILabelWiseLoss):
        pass


cdef extern from "cpp/losses/loss_label_wise_squared_error.h" namespace "boosting" nogil:

    cdef cppclass LabelWiseSquaredErrorLossImpl"boosting::LabelWiseSquaredErrorLoss"(ILabelWiseLoss):
        pass


cdef extern from "cpp/losses/loss_label_wise_squared_hinge.h" namespace "boosting" nogil:

    cdef cppclass LabelWiseSquaredHingeLossImpl"boosting::LabelWiseSquaredHingeLoss"(ILabelWiseLoss):
        pass


cdef class LabelWiseLoss:

    # Attributes:

    cdef shared_ptr[ILabelWiseLoss] loss_function_ptr


cdef class LabelWiseLogisticLoss(LabelWiseLoss):
    pass


cdef class LabelWiseSquaredErrorLoss(LabelWiseLoss):
    pass


cdef class LabelWiseSquaredHingeLoss(LabelWiseLoss):
    pass
