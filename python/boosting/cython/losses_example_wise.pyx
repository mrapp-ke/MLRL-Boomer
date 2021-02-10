"""
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)
"""
from libcpp.memory cimport make_shared


cdef class ExampleWiseLoss(Measure):
    """
    A wrapper for the pure virtual C++ class `IExampleWiseLoss`.
    """

    cdef shared_ptr[IMeasure] get_measure_ptr(self):
        return <shared_ptr[IMeasure]>self.loss_function_ptr


cdef class ExampleWiseLogisticLoss(ExampleWiseLoss):
    """
    A wrapper for the C++ class `ExampleWiseLogisticLoss`.
    """

    def __cinit__(self):
        self.loss_function_ptr = <shared_ptr[IExampleWiseLoss]>make_shared[ExampleWiseLogisticLossImpl]()

    def __reduce__(self):
        return (ExampleWiseLogisticLoss, ())
