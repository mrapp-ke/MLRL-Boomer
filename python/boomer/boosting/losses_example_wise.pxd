from libcpp.memory cimport shared_ptr


cdef extern from "cpp/losses_example_wise.h" namespace "boosting" nogil:

    cdef cppclass IExampleWiseLoss:
        pass


    cdef cppclass ExampleWiseLogisticLossImpl(IExampleWiseLoss):
        pass


cdef class ExampleWiseLoss:

    # Attributes:

    cdef shared_ptr[IExampleWiseLoss] loss_function_ptr


cdef class ExampleWiseLogisticLoss(ExampleWiseLoss):
    pass
