"""
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)
"""
from libcpp.memory cimport make_shared


cdef class ExampleWiseLoss:
    """
    A wrapper for the pure virtual C++ class `IExampleWiseLoss`.
    """
    pass


cdef class ExampleWiseLogisticLoss(ExampleWiseLoss):
    """
    A wrapper for the C++ class `ExampleWiseLogisticLoss`.
    """

    def __cinit__(self):
        self.loss_function_ptr = <shared_ptr[IExampleWiseLoss]>make_shared[ExampleWiseLogisticLossImpl]()
