"""
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)
"""
from libcpp.memory cimport make_shared


cdef class LabelWiseLoss(Measure):
    """
    A wrapper for the pure virtual C++ class `ILabelWiseLoss`.
    """

    cdef shared_ptr[IMeasure] get_measure_ptr(self):
        return <shared_ptr[IMeasure]>self.loss_function_ptr


cdef class LabelWiseLogisticLoss(LabelWiseLoss):
    """
    A wrapper for the C++ class `LabelWiseLogisticLoss`.
    """

    def __cinit__(self):
        self.loss_function_ptr = <shared_ptr[ILabelWiseLoss]>make_shared[LabelWiseLogisticLossImpl]()

    def __reduce__(self):
        return (LabelWiseLogisticLoss, ())


cdef class LabelWiseSquaredErrorLoss(LabelWiseLoss):
    """
    A wrapper for the C++ class `LabelWiseSquaredErrorLoss`.
    """

    def __cinit__(self):
        self.loss_function_ptr = <shared_ptr[ILabelWiseLoss]>make_shared[LabelWiseSquaredErrorLossImpl]()

    def __reduce__(self):
        return (LabelWiseSquaredErrorLoss, ())


cdef class LabelWiseSquaredHingeLoss(LabelWiseLoss):
    """
    A wrapper for the C++ class `LabelWiseSquaredHingeLoss`.
    """

    def __cinit__(self):
        self.loss_function_ptr = <shared_ptr[ILabelWiseLoss]>make_shared[LabelWiseSquaredHingeLossImpl]()

    def __reduce__(self):
        return (LabelWiseSquaredHingeLoss, ())
